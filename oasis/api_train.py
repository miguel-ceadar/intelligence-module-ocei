
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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# to be able to call `dataset/`
import bentoml
import numpy as np
import pandas as pd
import seaborn as sns
import logging
import mlflow
from pandas import DataFrame, concat
from tqdm import tqdm as tqdm_progress
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from dataset import root
from processing.utils import *
from analytics.dataframes import PersistentDF
from analytics.model_metrics import ModelMetricsDataClay
from tai.model_explainability import *
from tai.monitoring import *
from analytics.metrics import *
from models.management.registry import *
from models.arima.arima_compiler import *
from models.xgboost.xgboost_compiler import *
from models.pytorch.pytorch_compiler import *
from processing.process import *

bentoml_logger = logging.getLogger("bentoml")

class ModelTrain:
    def __init__(self, args):
        self.args = args
        self.metrics = {}
        self.results = {"Trained Model": "", "Dataset Splits": 0.2, "Results with test dataset": []}
        self.datasplit = {}
        # mlflow gc moved out of __init__ — see intelligence.trainers.mlflow_gc.
        # Constructor stays side-effect-free; the gc runs once per train session
        # at the top of initiate_train, preserving legacy cadence.

    def initiate_train(self):
        from intelligence.trainers import mlflow_gc
        bentoml_logger.info("Deleting mlruns stale folders")
        mlflow_gc()

        # Save the dataset split info
        if self.args.test_size:
            self.datasplit["train_data"] = (1 - self.args.test_size) * 100
            self.datasplit["test_data"] = self.args.test_size * 100
            bentoml_logger.info(f"Datasplit info: {self.datasplit}")

        dataset = self.args.dataset_name
        DATASET_PATH = os.path.join(root, dataset)
        bentoml_logger.info(f'Dataset: {DATASET_PATH}')
        model_type = self.args.model_type.lower()
        raw_data = load_data(self.args, bentoml_logger, DATASET_PATH)
        raw_data_clean = n_dimensional_dataset(self.args, raw_data)
        raw_data_clean = raw_data_clean.set_index('timestamp')
        train_df, test_df = data_simple_split(raw_data_clean, test_size=self.args.test_size) 
        """
        2 liner: unified dataset--> prepare_data inside of compilers 
        """
        if model_type != "pytorch":
            data_components = prepare_data(bentoml_logger, self.args, train_df, test_df, 
                                                   look_back=self.args.steps_back)
        else:
            data_components = prepare_data(bentoml_logger, self.args, train_df, test_df, 
                                           look_back=self.args.steps_back, 
                                           batch_size=self.args.batch_size)

        # Log metrics and parameters in mlflow
        bentoml_logger.info("MLFlow tracking")
        # Check mlruns if it exceeds above maximum limit then delete old runs
        try:
            mlruns = mlflow.search_runs()
            bentoml_logger.info(f"Mlflow runs: {mlruns}")
            if mlruns.run_id.count() > self.args.max_mlruns_count:
                bentoml_logger.info("Mlflow experiments repository has exceeded its limit")
                # Get run_id of oldest mlrun experiment
                run_id_old = mlruns.run_id.iloc[-1]
                delete_mlrun(bentoml_logger, run_id_old)
                bentoml_logger.info(f"Successfully deleted experiment: {run_id_old}")
            else:
                bentoml_logger.info("Mlflow experiments repository is within limit")
        except Exception as e:
            bentoml_logger.exception(e)

        model_handler = getattr(self, f'train_{model_type}_model', None)
        if model_handler:
            if model_type != "pytorch":
                return model_handler(data_components, train_df, test_df)
            return model_handler(data_components)  
        else:
            raise ValueError(f"Unsupported model type: {self.args.model_type}")

    def train_arima_model(self, data_components, train_df, test_df):
        return execute_arima(self.results, bentoml_logger, ModelMetricsDataClay, self.args, mlflow, bentoml, self.datasplit, data_components, train_df, test_df)

    def train_xgb_model(self, data_components, train_df, test_df):
        return execute_xgboost(self.results, bentoml_logger, ModelMetricsDataClay, self.args, mlflow, bentoml, self.datasplit, data_components, train_df, test_df)

    def train_pytorch_model(self, data_components):
        return execute_pytorch(self.results, bentoml_logger, ModelMetricsDataClay, self.args, mlflow, bentoml, self.datasplit, data_components)


import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Trainer')
    # default code
    parser.add_argument('--steps_back', default=6, help='Define how many steps to look back in a supervised time-series structure')
    parser.add_argument('--test_size', default=0.2, help='define datasplit size')
    parser.add_argument('--shap_samples', default=100, help='deafult sample size for shap')
    parser.add_argument('--max_mlruns_count', default=10, help='count for mlruns')

    parser.add_argument('--dataclay', default=False, help='True for using Dataclay, False otherwise')
    parser.add_argument('--dataclay_host', default="127.0.0.1", nargs='?', help='Define the host IP address')
    parser.add_argument('--dataclay_hostname', default="testuser", nargs='?', help='Define hostname')
    parser.add_argument('--dataclay_password', default="s3cret", nargs='?', help='Define password')
    parser.add_argument('--dataclay_dataset', default="testdata", nargs='?', help='Define dataset name')

    parser.add_argument('--dataset_name', default='cpu_sample_dataset_orangepi.csv', help='Name of the dataset filename to use')
    parser.add_argument('--model_type', default='XGB', choices=['XGB', 'ARIMA', 'PYTORCH'], help='Model type')
    parser.add_argument('--model_parameters', default={"xgboost_model_parameters": {"n_estimators": 1000, "max_depth": 7, "eta": 0.1, "subsample": 0.7,
                                                       "colsample_bytree": 0.8, "alpha": 0}}, help='Define model parameters')
    parser.add_argument('--num_variables', default=1, help='Define the number of variables you will read from the dataset')

    args = parser.parse_args()
    train = ModelTrain(args)
    results = train.initiate_train()
    print("returning... ",results)

