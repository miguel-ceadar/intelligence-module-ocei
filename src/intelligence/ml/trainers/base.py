"""``ModelTrainer`` — train ARIMA, XGBoost, and PyTorch-LSTM models from
precomputed ``data_components``.

Three independent methods (``train_arima`` / ``train_xgb`` /
``train_pytorch``); the ``ModelAdapter`` in ``intelligence.ml.models``
prepares the components dict and calls one. The data source (CSV /
PromQL / future OTel) is decoupled — see ``intelligence.telemetry``.
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

from intelligence.ml.trainers.lstm import LighterStudentLSTMModel, LSTMModel
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


def _make_loaders(train_dataset, test_dataset, batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


class ModelTrainer:
    """Train ARIMA / XGBoost / PyTorch-LSTM models from precomputed
    ``data_components``.

    Each ``train_*`` method is independent — call only the one for the
    model you want.
    """

    def __init__(self, data_components: dict[str, Any]) -> None:
        self.data_components = data_components

    def make_persistent(self, alias: str | None = None) -> None:
        # No-op kept for legacy-compiler call-site compatibility. Models
        # are persisted via ``bentoml.*.save_model`` inside each train_*
        # method below.
        return None

    # ---- XGBoost ------------------------------------------------------

    def train_xgb(self) -> tuple[dict, XGBRegressor]:
        model = XGBRegressor()
        model.set_params(**self.data_components["model_parameters"])
        logger.info("XGBRegressor: %s", model)

        model.fit(self.data_components["X_train"], self.data_components["y_train"])
        y_pred = model.predict(self.data_components["X_test"])
        y_pred = self.data_components["scaler_obj"].inverse_transform(y_pred.reshape(-1, 1))
        y_test = np.array(self.data_components["y_test"]).reshape(-1, 1)
        return compute_metrics(y_test, y_pred), model

    # ---- ARIMA --------------------------------------------------------

    def train_arima(self):
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
        return compute_metrics(y_test, y_pred), model, history, y_test, y_pred

    # ---- PyTorch LSTM -------------------------------------------------

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

    def _eval_lstm(self, model, val_loader, criterion, num_epochs, device):
        step_epoch: list[int] = []
        step_loss: list[float] = []
        for epoch in range(num_epochs):
            model.eval()
            eval_loss = 0.0
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                eval_loss += loss.item()
            avg = eval_loss / len(val_loader)
            step_epoch.append(epoch + 1)
            step_loss.append(avg)
            logger.info("Epoch [%d/%d], Eval Loss: %.4f", epoch + 1, num_epochs, avg)
        return model, step_epoch, step_loss

    def _kd_regression(
        self,
        teacher,
        student,
        train_loader,
        epochs,
        T,
        soft_target_loss_weight,
        ce_loss_weight,
        optimizer,
        method,
        device,
    ):
        torch.manual_seed(42)
        np.random.seed(42)
        teacher.eval()
        student.train()

        mse_criterion = nn.MSELoss()
        ce_criterion = nn.CrossEntropyLoss()

        step_epoch: list[int] = []
        step_loss: list[float] = []

        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.no_grad():
                    teacher_outputs = teacher(inputs)
                student_outputs = student(inputs)
                if method == 1:
                    loss = mse_criterion(student_outputs, teacher_outputs)
                else:
                    mse = mse_criterion(student_outputs, teacher_outputs)
                    soft = ce_criterion(student_outputs / T, teacher_outputs / T) * (T * T)
                    loss = soft_target_loss_weight * soft + ce_loss_weight * mse
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg = running_loss / len(train_loader)
            step_epoch.append(epoch + 1)
            step_loss.append(avg)
            if (epoch + 1) % 10 == 0:
                logger.info("Epoch [%d/%d], Loss: %.4f", epoch + 1, epochs, avg)

        return student, step_epoch, step_loss

    def _distilled_process(
        self,
        teacher_trained,
        train_loader,
        val_loader,
        input_size,
        output_size,
        num_epochs,
        criterion,
        device,
    ):
        hidden_size = 4
        student = LighterStudentLSTMModel(input_size, hidden_size, output_size).to(device)
        optimizer = optim.Adam(student.parameters(), lr=0.001)
        student_trained, se_s, sl_s = self._train_lstm(
            student, train_loader, criterion, optimizer, num_epochs, device
        )
        optimizer = optim.Adam(student.parameters(), lr=0.1)
        distilled, se_d, sl_d = self._kd_regression(
            teacher=teacher_trained,
            student=student_trained,
            train_loader=train_loader,
            epochs=50,
            T=2,
            soft_target_loss_weight=0.25,
            ce_loss_weight=0.75,
            optimizer=optimizer,
            method=2,
            device=device,
        )
        return distilled, se_s, sl_s, se_d, sl_d

    def train_pytorch(self, device: str = "cpu"):
        device_t = torch.device(device)
        params = self.data_components["model_parameters"]
        logger.info("Device: %s", device_t)
        logger.info("model_parameters: %s", params)

        input_size = params["input_size"]
        output_size = params["output_size"]
        hidden_size = params["hidden_size"]
        num_epochs = params["num_epochs"]
        distill = params.get("distill", False)

        batch_size = self.data_components["batch_size"]
        train_loader, val_loader = _make_loaders(
            self.data_components["train_dataset"],
            self.data_components["test_dataset"],
            batch_size=batch_size,
        )
        model = LSTMModel(input_size, hidden_size, output_size).to(device_t)
        logger.info("Floating point model: %s", model)
        print_size_of_model(model, "fp32")

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model, step_epoch_train, step_loss_train = self._train_lstm(
            model, train_loader, criterion, optimizer, num_epochs, device_t
        )
        model, _, _ = self._eval_lstm(model, val_loader, criterion, num_epochs, device_t)

        if distill:
            model, *_ = self._distilled_process(
                model,
                train_loader,
                val_loader,
                input_size,
                output_size,
                num_epochs,
                criterion,
                device_t,
            )

        outputs = model(self.data_components["X_test"].to(device_t))
        y_pred = outputs.cpu().detach().numpy().reshape(outputs.shape[0], outputs.shape[1])
        # y is shape (samples, horizon * num_variables). The scaler was fit on
        # (*, num_variables); flatten the horizon axis to make inverse_transform
        # shape-compatible, then restore.
        num_variables = int(self.data_components.get("num_variables", 1))
        samples = y_pred.shape[0]
        y_pred_inv = (
            self.data_components["scaler_obj"]
            .inverse_transform(y_pred.reshape(-1, num_variables))
            .reshape(samples, -1)
        )
        y_test_arr = self.data_components["y_test"]
        y_test_inv = (
            self.data_components["scaler_obj"]
            .inverse_transform(y_test_arr.reshape(-1, num_variables))
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
