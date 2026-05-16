"""``ModelTrainer`` — train ARIMA, XGBoost, and PyTorch-LSTM models from
precomputed ``data_components``. The per-kind ``Model`` classes in
``intelligence.ml.models`` prepare the components and call one method.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from statsmodels.tsa.arima.model import ARIMA
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from xgboost import XGBRegressor

from intelligence.ml.trainers.lstm import LSTMModel
from intelligence.ml.trainers.metrics import (
    metrics as compute_metrics,
)
from intelligence.ml.trainers.metrics import (
    metrics_pytorch,
    print_size_of_model,
)

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _make_train_loader(train_dataset, batch_size: int) -> DataLoader:
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


class ModelTrainer:
    """Train ARIMA / XGBoost / PyTorch-LSTM models from precomputed
    ``data_components``. Each ``train_*`` method is independent.
    """

    def __init__(self, data_components: dict[str, Any]) -> None:
        self.data_components = data_components

    def train_xgb(self) -> tuple[dict, XGBRegressor]:
        model = XGBRegressor()
        model.set_params(**self.data_components["model_parameters"])
        logger.info("XGBRegressor: %s", model)

        model.fit(self.data_components["X_train"], self.data_components["y_train"])
        y_pred = model.predict(self.data_components["X_test"])
        y_pred = self.data_components["scaler_obj"].inverse_transform(y_pred.reshape(-1, 1))
        y_test = np.array(self.data_components["y_test"]).reshape(-1, 1)
        return compute_metrics(y_test, y_pred), model

    def train_arima(self) -> tuple[dict, list]:
        """Walk-forward train + eval; returns ``(metrics, history)``.

        ``history`` is the scaled train+test series at the end of the
        walk, which the persisted artifact replays at predict time so
        new requests refit against the full observed window.
        """
        train = self.data_components["X_train"].squeeze(1)
        test = self.data_components["X_test"].squeeze(1)
        p = self.data_components["model_parameters"]["p"]
        d = self.data_components["model_parameters"]["d"]
        q = self.data_components["model_parameters"]["q"]

        history = list(train)
        predictions: list[float] = []

        for t in tqdm(range(len(test))):
            model = ARIMA(history, order=(p, d, q))
            model_fit = model.fit()
            yhat = float(model_fit.forecast()[0])
            predictions.append(yhat)
            history.append(test[t])

        y_pred = self.data_components["scaler_obj"].inverse_transform(
            np.array(predictions).reshape(-1, 1)
        )
        y_test = self.data_components["scaler_obj"].inverse_transform(
            self.data_components["X_test"].reshape(-1, 1)
        )
        return compute_metrics(y_test, y_pred), history

    def _train_lstm(self, model, train_loader, criterion, optimizer, num_epochs, device):
        step_epoch: list[int] = []
        step_loss: list[float] = []
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg = train_loss / len(train_loader)
            step_epoch.append(epoch + 1)
            step_loss.append(avg)
            logger.info("Epoch [%d/%d], Train Loss: %.4f", epoch + 1, num_epochs, avg)
        return model, step_epoch, step_loss

    def train_pytorch(self, device: str = "cpu"):
        device_t = torch.device(device)
        params = self.data_components["model_parameters"]
        logger.info("Device: %s", device_t)
        logger.info("model_parameters: %s", params)

        input_size = params["input_size"]
        output_size = params["output_size"]
        hidden_size = params["hidden_size"]
        num_epochs = params["num_epochs"]

        batch_size = self.data_components["batch_size"]
        train_loader = _make_train_loader(self.data_components["train_dataset"], batch_size)
        model = LSTMModel(input_size, hidden_size, output_size).to(device_t)
        logger.info("Floating point model: %s", model)
        print_size_of_model(model, "fp32")

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model, step_epoch_train, step_loss_train = self._train_lstm(
            model, train_loader, criterion, optimizer, num_epochs, device_t
        )

        outputs = model(self.data_components["X_test"].to(device_t))
        y_pred = outputs.cpu().detach().numpy().reshape(outputs.shape[0], outputs.shape[1])
        # y is shape (samples, horizon) — target-only output. ``scaler_obj``
        # is the target scaler (univariate); inverse-transform via a
        # column reshape and restore.
        samples = y_pred.shape[0]
        y_pred_inv = (
            self.data_components["scaler_obj"]
            .inverse_transform(y_pred.reshape(-1, 1))
            .reshape(samples, -1)
        )
        y_test_arr = self.data_components["y_test"]
        y_test_inv = (
            self.data_components["scaler_obj"]
            .inverse_transform(y_test_arr.reshape(-1, 1))
            .reshape(samples, -1)
        )

        out_metrics: dict = {}
        for i in range(y_test_inv.shape[1]):
            out_metrics[f"metric_{i}"] = metrics_pytorch(
                model, y_test_inv[:, i].reshape(-1, 1), y_pred_inv[:, i].reshape(-1, 1)
            )

        logger.info("Moving torch objects to CPU")
        model.cpu()
        return out_metrics, model, step_epoch_train, step_loss_train
