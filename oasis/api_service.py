"""
ICOS Intelligence Coordination
----------------------------------------
The ICOS Coordination API has two goals:
a) First, models can be pre-built and added to the API as specified in a Developer guide. The API outputs model predictions or information about a new model trained in this scenario. This is performed for easy integration of ML models with automated functions of the OS developed in ICOS.
b) Second, part of this API is targeted to extend ML libraries to make them available to a technical user to save storage resources in devices with access to the API. In this context, the API returns a framework environment to allow users easy plug-and-play with the environment already available in the API.

Copyright © 2022-2025 CeADAR Ireland

This program is free software: you can redistribute it and/or modify it under the terms of the Apache License 2.0 as published by the Free Software Foundation.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Apache License v2.0 for more details.
You should have received a copy of the Apache License v2.0 along with this program. If not, see <https://www.apache.org/licenses/LICENSE-2.0>.

This work has received funding from the European Union's HORIZON research
and innovation programme under grant agreement No. 101070177.
----------------------------------------
"""

# Importing the libraries
import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np
import bentoml
import requests
import logging
import torch
import os, json, uuid
from pydantic import BaseModel
from transformers import AutoTokenizer
from bentoml.io import NumpyNdarray, JSON, Text
from api_train import ModelTrain
import bentoml._internal.service.openapi as icos_api
from statsmodels.tsa.arima.model import ARIMA
from typing import Optional, Dict, Any
from tai.organizer_fl import OrganizerFL, Parameters
from tai.model_sync import ModelSyncService
from models.management.ai_model_repo import push as repo_push, pull as repo_pull

# Related to FL code from NKUA
TRIG_FILE = "/tmp/flwr_triggers/trigger.json"
TRIG_DIR = "/tmp/flwr_triggers" 
os.makedirs(TRIG_DIR, exist_ok=True)

# Bentoml logger init
bentoml_logger = logging.getLogger("bentoml")
# Bentoml API TAG init
icos_api.APP_TAG = icos_api.Tag(name="core", description="Meta kernel and Security related API service endpoints for training and prediction")

# Metrics Utilisation models
# Model 1
metrics_utilization_model_xgb = bentoml.sklearn.get("metrics_utilization_model_xgb:latest")
metrics_utilization_model_xgb_scaler = metrics_utilization_model_xgb.custom_objects["scaler_obj"]
metrics_utilization_model_xgb_driftobj = metrics_utilization_model_xgb.custom_objects["drift_detector_obj"]
metrics_utilization_model_xgb_runner = metrics_utilization_model_xgb.to_runner()

# Model 2
metrics_utilization_model_arima = bentoml.picklable_model.get("metrics_utilization_model_arima:latest")
metrics_utilization_model_arima_scaler = metrics_utilization_model_arima.custom_objects["scaler_obj"]
metrics_utilization_model_arima_drift_obj = metrics_utilization_model_arima.custom_objects["drift_detector_obj"]
metrics_utilization_model_arima_history_data = metrics_utilization_model_arima.custom_objects["historical_data"]
metrics_utilization_model_arima_runner = metrics_utilization_model_arima.to_runner()

# Model 3
metrics_utilization_model_lstm = bentoml.picklable_model.get("metrics_utilization_model_lstm:latest")
metrics_utilization_model_lstm_scaler = metrics_utilization_model_lstm.custom_objects["scaler_obj"]
metrics_utilization_model_lstm_runner = metrics_utilization_model_lstm.to_runner()

# Anomaly detection models
nkua_anomaly_detection = ["icos_nkua_clf_anomaly_detection_cell"+str(cellno)+":latest" for cellno in range(5)]
nkua_anomaly_detection_models = {model[:-7]:bentoml.sklearn.get(model) for model in nkua_anomaly_detection}
nkua_anomaly_detection_runner = [nkua_anomaly_detection_models[runner].to_runner() for runner in nkua_anomaly_detection_models]

# RNN LSTM models
nkua_rnn_lstm = ["icos_nkua_rnn_lstm_cell"+str(cellno)+":latest" for cellno in range(5)]
nkua_rnn_lstm_models = {model[:-7]:bentoml.keras.get(model) for model in nkua_rnn_lstm}
nkua_rnn_lstm_runner = [nkua_rnn_lstm_models[runner].to_runner() for runner in nkua_rnn_lstm_models]

# Energy consupmtion model
energy_consumption_model = bentoml.picklable_model.get("energy_consumption_forecast_xgb:latest")
energy_consumption_runner = energy_consumption_model.to_runner()

# Tetragon Security model
tetragon_security_model = bentoml.pytorch.get("security_model:latest")
tetragon_security_runner = tetragon_security_model.to_runner()

# ModelSync Instance
model_sync_service = ModelSyncService(check_interval=30)

# Load configuration file
with open("./api_service_configs.json") as file:
    api_config = json.load(file)

# Custom runner for ARIMA model
class ARIMAForecastRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self):
        self.arima = ARIMA

    @bentoml.Runnable.method(batchable=False)
    def forecast(self ,test_data, data_interrupt=False, saved_history_data=[]):
        predictions = []
        new_stream_history = []

        # Define the ARIMA tuning parameters
        p = api_config["model_parameters"]["arima_model_parameters"]["p"]  # AR order
        d = api_config["model_parameters"]["arima_model_parameters"]["d"]  # I order (degree of differencing)
        q = api_config["model_parameters"]["arima_model_parameters"]["q"]  # MA order

        # In case of device failure
        # train ARIMA with saved historical sample data
        if data_interrupt:
            bentoml_logger.info(f'Training with saved historical data: {saved_history_data}')
            history_data = saved_history_data
        else:
            history_data = metrics_utilization_model_arima.custom_objects["historical_data"]
        model = ARIMA(history_data, order=(p, d, q))
        model_fit = model.fit()
        output = model_fit.forecast()
        y_pred = output[0]
        predictions.append(y_pred)
        obs = test_data # single test value
        history_data.append(obs)
        new_stream_history = history_data[-api_config["history_sample_size"]:]
        bentoml_logger.info(f'New history {new_stream_history[-5:]}')
        bentoml_logger.info(f'History length {len(history_data)}')

        # Delete bentoml models if repo exceeds more than 5 models
        # Get models list from current bento
        models_list = bentoml.models.list('metrics_utilization_model_arima')
        bentoml_logger.info(f'Number of models: {len(models_list)}')
        bentoml_logger.info(f'Get bentoml models list {models_list}')
        # Try to get models list using creating time and dataframe
        # move it to yaml file
        if len(models_list) > 5:
            bentoml_logger.info('Bentoml models repository has exceeded its limit')
            delete_bentoml_model(models_list)
        else:
            bentoml_logger.info('Bentoml models repository is within limit')
        # Save model with BentoML
        bentoml.picklable_model.save_model('metrics_utilization_model_arima',
                                                        model,
                                                        custom_objects= {"scaler_obj": metrics_utilization_model_arima_scaler,
                                                                         "historical_data": new_stream_history,
                                                                         "drift_detector_obj": metrics_utilization_model_arima.custom_objects["drift_detector_obj"],
                                                                         "model_metrics": metrics_utilization_model_arima.custom_objects["model_metrics"],
                                                                         "test_sample_size": metrics_utilization_model_arima.custom_objects["test_sample_size"]},
                                                        signatures={"predict": {"batchable": True}},
        )
        return y_pred

# Initialise custom arima runner
model_arima_custom_runner = bentoml.Runner(ARIMAForecastRunnable)

# Initialise service
svc = bentoml.Service(
    name="analytics",
    runners=[
        metrics_utilization_model_xgb_runner,
        metrics_utilization_model_arima_runner,
        model_arima_custom_runner,
        metrics_utilization_model_lstm_runner,
        nkua_anomaly_detection_runner[0],
        nkua_anomaly_detection_runner[1],
        nkua_anomaly_detection_runner[2],
        nkua_anomaly_detection_runner[3],
        nkua_anomaly_detection_runner[4],
        nkua_rnn_lstm_runner[0],
        nkua_rnn_lstm_runner[1],
        nkua_rnn_lstm_runner[2],
        nkua_rnn_lstm_runner[3],
        nkua_rnn_lstm_runner[4],
        energy_consumption_runner,
        tetragon_security_runner,
    ],
    )
class FederatedControl(BaseModel):
    run_config: Optional[Dict[str, Any]] = None  # for dynamic CLI args
    use_stream: bool = True                      # whether to add `--stream`

class TrainFeatures(BaseModel):
    # Default values
    model_name: str = api_config["model_name"]
    model_type: str = api_config["model_type"]
    test_size: float = api_config["test_size"]
    dataset_name: str = api_config["dataset_name"]
    num_variables: int = api_config["num_variables"]
    steps_back: int = api_config["steps_back"]
    batch_size: int = api_config["batch_size"]
    max_models_count: int = api_config["max_models_count"]
    max_mlruns_count: int = api_config["max_mlruns_count"]
    shap_samples: int = api_config["shap_samples"]
    dataclay: bool = api_config["dataclay"]
    dataclay_host: str = api_config["dataclay_host"]
    dataclay_hostname: str = api_config["dataclay_hostname"]
    dataclay_password: str = api_config["dataclay_password"]
    dataclay_dataset: str = api_config["dataclay_dataset"]
    model_parameters: dict = api_config["model_parameters"]
    device: str = api_config["device"]

    # Federated
    use_federated: bool = api_config["federated"]["use_federated"]
    federated_params: Optional[FederatedControl] = FederatedControl(
        run_config=api_config["federated"].get("run_config", {}),
        use_stream=api_config["federated"].get("use_stream", True)
    )


class TrainOutput(BaseModel):
    results: dict

class PredictFeatures(BaseModel):
    # Default values
    model_tag: str = api_config["model_tag"]
    metric_type: int = api_config["metric_type"]
    steps_back: int = api_config["steps_back"]
    input_series: dict = api_config["input_series"]
    history_sample_size: int = api_config["history_sample_size"]
    data_interruption: bool = api_config["data_interruption"]
    history_data: list = api_config["history_data"]

class PredictOutput(BaseModel):
    results: dict

# Train features
input_train = JSON(pydantic_model=TrainFeatures)
output_train = JSON(pydantic_model=TrainOutput)
# Predict features
input_predict = JSON(pydantic_model=PredictFeatures)
output_predict = JSON(pydantic_model=PredictOutput)

# Delete models from bentoml repo if exceeds limit
def delete_bentoml_model(models_list):
    try:
        # Create data dictionary of models list
        data = [model.info.to_dict() for model in models_list]
        model_df = pd.DataFrame(data)
        model_df_sorted = model_df.sort_values('creation_time').iloc[0]
        model_name = model_df_sorted['name']
        model_tag = model_df_sorted['version']
        delete_model = model_name + ':' + model_tag
        bentoml_logger.info(f"Model to be deleted: {delete_model}")
        bentoml.models.delete(delete_model)
        bentoml_logger.info(f"Succesfully deleted model: {delete_model}")
    except Exception as e:
            raise Exception(e,sys)

# Generate model confidence for predictions
def model_confidence(prediction, error, sample_size, alpha=1.96):
    # Here we will be using Mean Absolute Error (MAE)
    e = error
    if e < 6:
        bentoml_logger.info("Model confidence is high and suitable for prediction")
    else:
        bentoml_logger.warning("Model confidence is low and not suitable for prediction")
    # Calculate confidence score
    confidence = 100 - e
    n = sample_size
    yhat_out = prediction
    # Calculate the confidence interval
    margin_of_error = alpha * (abs(e * (1 - e) / n)**0.5)
    lower_bound = yhat_out - margin_of_error
    upper_bound = yhat_out + margin_of_error
    bentoml_logger.info(yhat_out)
    bentoml_logger.info(f'Model Confidence: {confidence:.2f}% with 95% Prediction Interval: {lower_bound:.3f} to {upper_bound:.3f}')
    return confidence, lower_bound, upper_bound

# Trigger ICOS-FL
# one‑liner to fetch or create the singleton organiser
def _organizer() -> OrganizerFL:
    try:
        return OrganizerFL.get_by_alias("global_organizer_fl")
    except Exception:
        org = OrganizerFL()
        org.make_persistent(alias="global_organizer_fl")
        return org
    
def run_federated_learning(fed_params: FederatedControl):
    # ------------------ build the trigger object --------------------------- #
    params = Parameters(
        action="start",
        use_stream=True,                  # the old “--stream” flag
        run_config=fed_params.run_config  # the dict you already receive
    )

    # ------------------ enqueue into DataClay ------------------------------ #
    _organizer().new_trigger(params)

    # bentoml_logger.info("FL trigger enqueued via DataClay: %s", params.__dict__)

    # Because Parameters is a DataClayObject, it has certain internal attributes that
    # we cannot serialize / we do not need in the API. Just keep the relevant ones.
    serializable_params = {
        "action": params.action,
        "use_stream": params.use_stream,
        "run_config": params.run_config
    }
    return {
        "status": "Federated training requested",
        "trigger": serializable_params,
    }
# Train API service
@svc.api(input=input_train, output=output_train)
def train(input_data:TrainFeatures):
    '''
    API to train a model to predict metrics utilisation

    ==================
    Input Parameters:
    ==================

    * model_name <str>: Name of the model train file
    * model_type <str>: Specify the model type to be used; XGBoost or ARIMA
    * test_size <float>: To define Test size for dataset split
    * dataset_name <str>: Specify dataset name to be used to train the model
    * steps_back <int>: To define number of past CPU values for the model to predict a future CPU value
    * model_parameters <dic>: To define model parameters used for model training
    * max_models_count <int>: Specify the maximum number of trained models to keep in the repository
    * max_mlruns_count <int>: Specify the maximum number of mlflow model runs for model tracking
    * shap_samples <int>: Specify the number of samples for Shap to generate explainability
    * device <str>: Specify device (cpu or gpu) to run the training workload
    * dataclay <bool>: Set to True if model training needs to run using dataclay or else False
    * dataclay_host <str>: Specify dataclay host ip
    * dataclay_hostname <str>: Specify dataclay host name
    * dataclay_password <str>: Specify dataclay password
    * dataclay_dataset <str>: Specify dataclay dataset

    ==================
    Output Parameters:
    ==================

    * results <dict>: Returns a dictionary of values
        * Trained model: Tag of the trained model saved in BentoML local repository
        * Dataset split: Dataset split size, train and test
        * Metrics: Trained model metrics with the test data
    '''

    bentoml_logger.info("Inside train data service")

    # Federated learning block
    if input_data.use_federated:
        from intelligence.adapters import DataClayConfig, dataclay_client
        if not input_data.federated_params:
            raise ValueError("Federated training is enabled, but no parameters were provided.")
        fl_dc_config = DataClayConfig(
            enabled=True,
            proxy_host=input_data.dataclay_host,
            username=input_data.dataclay_hostname,
            password=input_data.dataclay_password,
            dataset=input_data.dataclay_dataset,
        )
        with dataclay_client(fl_dc_config):
            bentoml_logger.info("Federated learning activated")
            results = run_federated_learning(input_data.federated_params)
        return {"results": results}

    # Classic training
    if input_data.model_type in {"XGB", "ARIMA", "PYTORCH"}:
        train = ModelTrain(input_data)
        bentoml_logger.info(f"Training initiated with model_type: {input_data.model_type}")
        results = train.initiate_train()
        bentoml_logger.info(results)

        # Prune models if count exceeds max
        model_key_map = {
            "XGB": "metrics_utilization_model_xgb",
            "ARIMA": "metrics_utilization_model_arima",
            "PYTORCH": "metrics_utilization_model_lstm"
        }
        model_key = model_key_map.get(input_data.model_type)
        models_list = bentoml.models.list(model_key)
        if len(models_list) > input_data.max_models_count:
            bentoml_logger.warning("Model repo exceeds limit, pruning...")
            delete_bentoml_model(models_list)
        else:
            bentoml_logger.info("Model repo is within safe limits")
        return results
    else:
        raise ValueError(f"Unsupported model_type: {input_data.model_type}")

# Predict API service
@svc.api(input=input_predict, output=output_predict)
def predict(input_data: PredictFeatures):
    '''
    API to generate predicitions

    ==================
    Input Parameters:
    ==================

    * model_tag <str>: Specify the trained model tag name to be used for prediction
    * metric_type <str>: Specify the type of metric to be used for telemetry API
    * steps_back <int>: Specify the number of past values to which the model has been trained for predicting one step in future
    * input_series <list>: Provide past metrics values(lags) in time ascending order. For instance, if input series is [t-6, t-5, t-4, t-3, t-2, t-1], then we predit 't'.
    * history_sample_size <int>: Sample size to be used to train ARIMA model.
    * data_interruption <boolean>: If device was interruted due to some technical issue, set it to true else false.
    * history_data <list>: If data_interruption is true, provide history_data of 500 samples to train ARIMA

    ==================
    Output Parameters:
    ==================

    * result <dict>: Returns a dictionary of values
        * Model prediction: Predicted value
        * Confidence Score: Model confidence score for the prediction
        * Confidence intervals: Lower and upper bounds of the predictions
    '''
    result = {}
    bentoml_logger.info("Inside predict data service")
    bentoml_logger.info(f"Input series: {input_data.input_series}")
    try:
        # Choose which model type to be used for inferencing
        if 'xgb' in input_data.model_tag and 'metrics_utilization' in input_data.model_tag:
            bentoml_logger.info(f"Predicting 1-step future value from {input_data.steps_back} input sequence of values")
            # for univariate model
            data_dict = {"TARGET": input_data.input_series['input_1']}
            data = pd.DataFrame(data=data_dict)
            model_xgb = bentoml.sklearn.get(input_data.model_tag)
            bentoml_logger.info(f"Using trained model {input_data.model_tag} from bentoml local store")
            model_xgb_scaler = model_xgb.custom_objects["scaler_obj"]
            # model_xgb_driftobj = model_xgb.custom_objects["drift_detector_obj"]
            model_xgb_runner = model_xgb.to_runner(embedded=True)
            model_xgb_runner.init_local(quiet=True)
            # Preprocess the testing data
            input_data_scaled = model_xgb_scaler.transform(data.values)
            # Predict the next CPU value for the testing data
            output_data_scaled = model_xgb_runner.run(input_data_scaled.reshape(1,-1))
            # Inverse scale the predicted values
            prediction = model_xgb_scaler.inverse_transform(output_data_scaled.reshape(-1,1))
            # Calculate prediction interval
            error = model_xgb.custom_objects["model_metrics"]
            confidence, lower_bound, upper_bound = model_confidence(prediction=prediction[0][0],
                                                        error=error["mae"],
                                                        sample_size=model_xgb.custom_objects["test_sample_size"])
            result["model_prediction"] = round(prediction[0][0],3)
            result["model_confidence"] = confidence
            result["95%_confidence_interval"] = str(round(lower_bound,3))+ " to " + str(round(upper_bound,3))

        elif 'arima' in input_data.model_tag and 'metrics_utilization' in input_data.model_tag:
            bentoml_logger.info(f"Predicting 1-step future value from {input_data.steps_back} input sequence of values")
            # for univariate model
            data_dict = {"TARGET": input_data.input_series['input_1']}
            data = pd.DataFrame(data=data_dict)
            model_arima = bentoml.picklable_model.get(input_data.model_tag)
            bentoml_logger.info(f"Using trained model {input_data.model_tag} from bentoml local store")
            model_arima_scaler = model_arima.custom_objects["scaler_obj"]
            model_arima_runner = model_arima.to_runner(embedded=True)
            model_arima_runner.init_local(quiet=True)
            # Preprocess the testing data
            input_data_scaled = model_arima_scaler.transform(data.values)
            # Predict the next CPU value for the testing data
            # For data loss due to device interruption
            if input_data.data_interruption:
                bentoml_logger.warn('Device interrupted!!!')
                bentoml_logger.warn("Data loss due to device or network failure")
                predicted_data_scaled = model_arima_custom_runner.forecast.run(input_data_scaled[0][0],
                                                                    data_interrupt=True,
                                                                    saved_history_data=input_data.history_data)
            else:
                predicted_data_scaled = model_arima_custom_runner.forecast.run(input_data_scaled[0][0])
            # Inverse scale the predicted values
            prediction = model_arima_scaler.inverse_transform(np.array(predicted_data_scaled).reshape(-1,1))
            # Calculate prediction interval
            error = model_arima.custom_objects["model_metrics"]
            confidence, lower_bound, upper_bound = model_confidence(prediction=prediction[0][0],
                                                        error=error["mae"],
                                                        sample_size=model_arima.custom_objects["test_sample_size"])
            result["model_prediction"] = round(prediction[0][0],3)
            result["model_confidence"] = confidence
            result["95%_confidence_interval"] = str(round(lower_bound,3))+ " to " + str(round(upper_bound,3))

        elif 'lstm' in input_data.model_tag and 'metrics_utilization' in input_data.model_tag:
            # Dynamically generate DataFrame from all input_series
            data_dict = {f"column_{idx}": input_data.input_series[key]
                        for idx, key in enumerate(input_data.input_series)}
            data = pd.DataFrame(data=data_dict)

            # Load model and scaler
            model_lstm = bentoml.picklable_model.get(input_data.model_tag)
            bentoml_logger.info(f"Using trained model {input_data.model_tag} from bentoml local store")

            model_lstm_scaler = model_lstm.custom_objects["scaler_obj"]
            model_lstm_runner = model_lstm.to_runner(embedded=True)
            model_lstm_runner.init_local(quiet=True)

            # Scale inputs
            input_data_scaled = model_lstm_scaler.transform(data.values)

            # Expand dimensions for input: (1, sequence_len, num_features)
            input_data_scaled = np.expand_dims(input_data_scaled, axis=0)
            input_data_scaled = torch.from_numpy(input_data_scaled).float()

            # Model prediction
            output_data_scaled = model_lstm_runner.run(input_data_scaled)
            output_data_scaled = output_data_scaled.detach().cpu().numpy()

            # Inverse scale the predictions
            prediction = model_lstm_scaler.inverse_transform(output_data_scaled.reshape(1, -1))

            # Generate per-feature metrics
            for i in range(prediction.shape[1]):
                try:
                    error_metrics = model_lstm.custom_objects[f"model_metrics_{i+1}"]
                except KeyError:
                    # Fall back to generic metric name if old format
                    error_metrics = model_lstm.custom_objects.get(f"metric_{i}", {})

                confidence, lower, upper = model_confidence(
                    prediction=prediction[0][i],
                    error=error_metrics.get("mae", 0),
                    sample_size=model_lstm.custom_objects["test_sample_size"]
                )

                result[f"metric_{i}"] = {
                    "model_prediction": round(prediction[0][i], 3),
                    "model_confidence": confidence,
                    "95%_confidence_interval": f"{round(lower,3)} to {round(upper,3)}"
                }
            result["metric_type"] = input_data.metric_type

        elif 'anomaly' in input_data.model_tag and 'nkua' in input_data.model_tag:
            bentoml_logger.info("Inside anomaly detection clasification service")
            # Predict data using trained model
            cell_number = int(input_data.model_tag.split(":")[0][-1])
            bentoml_logger.info(f"cell_number: {cell_number}")
            result["model_prediction"] = nkua_anomaly_detection_runner[cell_number].run(input_data.input_series['input_1'])

        elif 'lstm' in input_data.model_tag and 'nkua' in input_data.model_tag:
            bentoml_logger.info("Inside rnn lstm clasification service")
            bentoml_model = bentoml.keras.get(input_data.model_tag)
            cell_number = int(input_data.model_tag.split(":")[0][-1])
            bentoml_logger.info(f"cell_number: {cell_number}")
            # Predict data using trained model
            output_data_scaled = nkua_rnn_lstm_runner[cell_number].run(input_data.input_series['input_1'])
            # Inverse scale the predicted values
            nkua_rnn_lstm_scaler = bentoml_model.custom_objects["scaler"]
            result["model_prediction"] = nkua_rnn_lstm_scaler.inverse_transform(output_data_scaled)

        elif 'energy_consumption' in input_data.model_tag:
            bentoml_logger.info("Energy consumption model service")
            # Predict data using trained model
            prediction = energy_consumption_runner.run(input_data.input_series['input_1'])
            result["model_prediction"] = prediction

        elif 'security_model' in input_data.model_tag:
            bentoml_logger.info("Tetragon security model service")
            # Load pre-trained model and tokenizer
            model_name = "u-haru/log-inspector"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            input = tokenizer(input_data.input_series['input_1'], return_tensors="pt", padding=True, truncation=True, max_length=512)
            output = tetragon_security_runner.run(**input)
            predicted_class = torch.argmax(output[0], dim=-1)
            result["model_prediction"] = predicted_class.item()
        
        # Case where a model is pushed to the repo from Huggingface repo
        else:
            # for univariate model
            data_dict = {"TARGET": input_data.input_series['input_1']}
            data = pd.DataFrame(data=data_dict)
            # Check if the model is found inside the AI repo
            model_name = input_data.model_tag.split(":")[0]
            model_version = input_data.model_tag.split(":")[1]
            model_list = bentoml.models.list(input_data.model_tag)
            if not model_list:
                bentoml_logger.info(f"New Model found inside AI repo: {model_list[0]}")
                # Load model and scaler
                load_model = bentoml.models.get(input_data.model_tag)
                bentoml_logger.info(f"Using trained model {input_data.model_tag} from bentoml local store")
                model_scaler = load_model.custom_objects["scaler"]
                if model_scaler:
                    # Scale inputs
                    input_data_scaled = model_scaler.transform(data.values)
                model_runner = load_model.to_runner(embedded=True)
                model_runner.init_local(quiet=True)
                # Predict with scaled data
                output_data_scaled = model_runner.run(input_data_scaled.reshape(1,-1))
                prediction = model_scaler.inverse_transform(output_data_scaled.reshape(-1,1))
                result["model_prediction"] = round(prediction[0][0],3)
            else:
                bentoml_logger.exception("No model found in the repo")
                result["model_prediction"] = []


        result["metric_type"] = input_data.metric_type

        bentoml_logger.info(result)

        return result

    except Exception as e:
        raise Exception(e,sys)
    
# Detect drift -  API service
@svc.api(input=input_predict, output=NumpyNdarray(dtype="bool"))
def detect_drift(input_data: PredictFeatures):
    bentoml_logger.info("Inside Drift-detection service")
    bentoml_logger.info(f"Input series (analysis data): {input_data.input_series}")
    bentoml_logger.info(f"Analizing drifts on analysis data")
    try:
        # Create data-frame from input series
        if len(input_data.input_series) > 1:
            # for multivariate scenario with two model outputs
            data_dict = {"CPU": input_data.input_series['input_1'],
                         "MEM": input_data.input_series['input_2']}
        else:
            # for univariate model
            data_dict = {"TARGET": input_data.input_series['input_1']}

        data = pd.DataFrame(data=data_dict)

        # Choose which model type to be used for inferencing
        if 'xgb' in input_data.model_tag:
            metrics_utilization_model_xgb = bentoml.sklearn.get(input_data.model_tag)
            bentoml_logger.info(f"Using trained model {input_data.model_tag} from bentoml local store")
            metrics_utilization_model_xgb_scaler = metrics_utilization_model_xgb.custom_objects["scaler_obj"]
            # Preprocess the testing data
            input_data_scaled = metrics_utilization_model_xgb_scaler.transform(data.values)
            # transform data to nannyml format timestamp, pred, gt
            analysis_df = {'y_pred': input_data_scaled.squeeze().astype(np.float64)}
            analysis_df = pd.DataFrame(analysis_df, dtype='object')

        elif 'arima' in input_data.model_tag:
            metrics_utilization_model_arima = bentoml.picklable_model.get(input_data.model_tag)
            bentoml_logger.info(f"Using trained model {input_data.model_tag} from bentoml local store")
            metrics_utilization_model_arima_scaler = metrics_utilization_model_arima.custom_objects["scaler_obj"]
            # Preprocess the testing data
            input_data_scaled = metrics_utilization_model_arima_scaler.transform(data.values)
            # transform data to nannyml format timestamp, pred, gt
            analysis_df = {'y_pred': input_data_scaled.squeeze().astype(np.float64)}
            analysis_df = pd.DataFrame(analysis_df, index=[0])

        ######### Drift-detector ###########
        # transform data to nannyml format timestamp, pred, gt
        analysis_df.reset_index(drop=True, inplace=True)

        bentoml_logger.info(f"analysis_df {analysis_df}")
        # Calculate drift
        if 'xgb' in input_data.model_tag:
            results_nannyml = metrics_utilization_model_xgb_driftobj.calculate(analysis_df)
        elif 'arima' in input_data.model_tag:
            results_nannyml = metrics_utilization_model_arima_drift_obj.calculate(analysis_df)
        res_analysis = results_nannyml.filter(period='analysis').to_df()
        bentoml_logger.info(f"NannyML table {res_analysis}")
        alert_columns = [('y_pred', 'jensen_shannon','alert')]
        alert_mask = res_analysis[alert_columns].any(axis=1)
        result = alert_mask.any()
        bentoml_logger.info(f"[ALERT] - Drift detection: {result}")
        return result
    except Exception as e:
        raise Exception(e,sys)


# LOMOS input features
class LomosFeatures(BaseModel):
    # Default values
    api_url: str = api_config["api_url"]
    api_params: dict = api_config["api_params"]

input_lomos = JSON(pydantic_model=LomosFeatures)

# LOMOS API service
@svc.api(input=input_lomos, output=JSON())
def get_anomalies(input_data: LomosFeatures):
    '''
    LOMOS API to get anomalies

    ==================
    Input Parameters:
    ==================

    GET REQUEST
    * anomaly_type - mention type of anomaly
    * from_timestamp - return only anomalies older than
    * to_timestamp - return only anomalies jounger than
    * min_anomaly_score - return only anomlies with anomaly score >=
    * maybe also max_count limit, or pagesize

    For Instance:


    curl "http://lomos_ip/api/anomalies?from=20231220T10:00:00.000Z&to=20231220T11:00:00.000Z&min_anomaly_score=0.8"
    curl "http://10.160.3.227:25001/api/top_anomaly?min_anomaly_score=0.1&from_timestamp=2024-01-08T00:00:01.200000Z&to_timestamp=2024-01-18T00:00:01.200000Z"

    curl "http://10.160.3.227:25001/api/top_anomaly?min_anomaly_score=0.1&from_timestamp=2024-01-08T00:00:01.200000Z&to_timestamp=2024-01-18T00:00:01.200000Z"

    ==================
    Output Parameters:
    ==================

    GET REPLY (JSON)
    * aggregate:
        - count - total count of matching anomalies/messages
        - min_anomaly_score - lowest anomaly score of all messages
        - max_anomaly_score - highest anomaly score of all messages
    * messages - list of messages, each message is
        - source - something to identify the source system, e.g. VM or container name (or IP, if available).
        - timestamp
        - anomaly_score - 0.0 to 1.0, 0 is very normal, 1 is very abnormal
        - log_line - original line from log file

    '''
    bentoml_logger.info("Inside Lomos API service")
    bentoml_logger.info("Generating GET request")
    bentoml_logger.info(f"LOMOS API parameters:{input_data.api_params}")
    response = requests.get(input_data.api_url, params=input_data.api_params)
    bentoml_logger.info(f"Response from LOMOS API: {response.url}")
    return response.json()


# MLFlow input features
class MlflowFeatures(BaseModel):
    # Default values
    mlflow_port: str = api_config["mlflow_port"]
    mlflow_host: str = api_config["mlflow_host"]

input_mlflow = JSON(pydantic_model=MlflowFeatures)

# MLFlow API service
@svc.api(input=input_mlflow, output=JSON())
def launch_mlflow_ui(input_data: MlflowFeatures):
    '''
    Use this request to start MlFlow UI service

    ==================
    Input Parameters:
    ==================

    * mlflow port <str>: Specify mlflow port (default: 5000)
    * mlflow host <str>: Specify mlflow host ip (default: 127.0.0.1)

    ==================
    Output Parameters:
    ==================

    * Retuns a string of hostip and port for mlflow ui to launch
    '''
    subprocess.Popen(['mlflow', 'server', '--host', input_data.mlflow_host, '--port', input_data.mlflow_port])
    return {"mlflow ui started at": "http:"+input_data.mlflow_host+":"+input_data.mlflow_port}


# Show models input features
class ShowModels(BaseModel):
    # Default values
    model: str = api_config["model"]

input_show_models = JSON(pydantic_model=ShowModels)
# List models
@svc.api(input=input_show_models, output=JSON())
def show_models(input_data: ShowModels):
    '''
    Use this request to list models from the repository
    ==================
    Input Parameters:
    ==================

    * model <str>: Specify model name to show list (default: all)

    ==================
    Output Parameters:
    ==================

    * result <dict>: Returns a dictionary of values
        * models_count: Number of models in the repo
        * models_list: Name and version of models in the repo
    '''
    result = {}
    bentoml_logger.info(f"Input text: {input_data.model}")
    if input_data.model == "all":
        models_list = bentoml.models.list()
    else:
        try:
            models_list = bentoml.models.list(input_data.model)
        except Exception as e:
            raise Exception(e,sys)
    bentoml_logger.info(f"Model list: {np.array(models_list)}")
    result["models_count"] = len(models_list)
    result["models_list"] = {x.tag.version:x.tag.name for x in models_list}
    return result

# Delete a model from registry
class RemoveModel(BaseModel):
    model_tag: str = api_config["model_tag"]
input_remove_model = JSON(pydantic_model=RemoveModel)

@svc.api(input=input_remove_model, output=JSON())
def remove_model(input_data:RemoveModel):
    '''
    Use this request to remove a model from the registry
    ==================
    Input Parameters:
    ==================

    * model_tag <str>: Specify model Tag to remove from the registry

    ==================
    Output Parameters:
    ==================

    * result <bool>: Returns bool value True when model is deleted
    '''
    result = {}
    bentoml_logger.info(f"Removing model: {input_data.model_tag}")
    try:
        # Get info incase 'latest' is used as model_tag
        model_info = bentoml.models.get(input_data.model_tag)
        bentoml.models.delete(input_data.model_tag)
        bentoml_logger.info(f"Succesfully deleted model: {model_info}")
        result["Removed_model"] =  {"name": model_info.tag.name,
                "version":model_info.tag.version}
    except Exception as e:
        bentoml_logger.exception(f"Failed to delete model due an error")
        raise Exception(e,sys)
    return result

class ModelRepoSync(BaseModel):
    push: bool = api_config["model_repo"].get("push", False)
    pull: bool = api_config["model_repo"].get("pull", False)

    hf_token: str = api_config["model_repo"]["hf_token"]
    repo_id:  str = api_config["model_repo"]["repo_id"]
    model_tag: str = api_config["model_repo"]["model_tag"]

    commit_message: str = api_config["model_repo"].get("commit_message", "Sync model")

input_model_repo_sync  = JSON(pydantic_model=ModelRepoSync)
output_model_repo_sync = JSON()

@svc.api(input=input_model_repo_sync, output=output_model_repo_sync)
def sync_model_repo(input_data: ModelRepoSync):
    """
    Push and/or pull a BentoML model to / from a Hugging Face repo,
    depending on the `push` / `pull` flags in the request (or defaults
    from `api_service_configs.json`).
    """
    if not (input_data.push or input_data.pull):
        raise ValueError("At least one of `push` or `pull` must be true.")

    response = {
        "status": "success",
        "repo_id": input_data.repo_id,
        "model_tag": input_data.model_tag,
        "actions": []
    }

    if input_data.push:
        bentoml_logger.info(f"Pushing {input_data.model_tag} → {input_data.repo_id}")
        repo_push(
            bentoml_model_tag=input_data.model_tag,
            repo_id=input_data.repo_id,
            hf_token=input_data.hf_token,
            commit_message=input_data.commit_message,
        )
        response["actions"].append("push")

    if input_data.pull:
        bentoml_logger.info(f"Pulling {input_data.model_tag} from {input_data.repo_id}")
        local_path = repo_pull(
            repo_id=input_data.repo_id,
            bentoml_model_tag=input_data.model_tag,
            hf_token=input_data.hf_token,
        )
        response["actions"].append("pull")
        response["local_path"] = local_path 

    models_list = bentoml.models.list()
    response["models_count"] = len(models_list)
    response["models_list"] = {x.tag.version: x.tag.name for x in models_list}

    return response


@svc.on_startup
def startup(context):
    """Initialize services on API startup."""
    try:
        # Start model sync service in background
        model_sync_service.start()
        bentoml_logger.info("Model Sync Service started successfully ✓")
    except Exception as e:
        bentoml_logger.error(f"Failed to start Model Sync Service: {e}")

@svc.on_shutdown  
def shutdown(context):
    """Cleanup services on API shutdown."""
    try:
        model_sync_service.stop()
        bentoml_logger.info("Model Sync Service stopped ✓")
    except Exception as e:
        bentoml_logger.error(f"Error stopping Model Sync Service: {e}")
