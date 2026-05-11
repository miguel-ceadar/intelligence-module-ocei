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

from analytics.metrics import *
from processing.utils import *

from analytics.dataframes import PersistentDF
from intelligence.adapters import dataclay_client


def load_data(args, bentoml_logger, dataset_path):
    if args.dataclay:
        bentoml_logger.info("Collecting Raw data to/from dataclay")
        with dataclay_client(args):
            metricsdf = PersistentDF(dataset_path)
            metricsdf.make_persistent(alias=args.dataset_name)
            bentoml_logger.info("Read dataset as a dataframe")
            metrics_dataset = PersistentDF.get_by_alias(args.dataset_name)
            raw_data = pd.read_csv(metrics_dataset.content)
    else:
        bentoml_logger.info("Collecting raw data from local device")
        raw_data = pd.read_csv(dataset_path)
    return raw_data


def prep_data_components(X_train=None, X_test=None, y_train=None, y_test=None, scaler_obj=None, 
                         model_params=None, train_dataset=None, test_dataset=None, batch_size=None):
       data_components = {}
       data_components['X_train']=X_train
       data_components['X_test']=X_test
       data_components['y_train']=y_train
       data_components['y_test']=y_test
       data_components['scaler_obj']=scaler_obj
       data_components['model_parameters'] = model_params
       data_components['train_dataset'] = train_dataset
       data_components['test_dataset']  = test_dataset
       data_components['batch_size']  = batch_size
       return data_components

def n_dimensional_dataset(args, raw_data):
    
    raw_data = raw_data.iloc[:, :args.num_variables+1]# args.num_variables+1 considering timestap as an additional
    new_column_names = ['timestamp'] + [f'column_{i}' for i in range(1, args.num_variables+1)]
    raw_data.columns = new_column_names
    
    raw_data.replace("undefined", np.nan, inplace=True)
    for col in raw_data.columns:
            if col != "timestamp":
                raw_data[col] = pd.to_numeric(raw_data[col], errors='coerce')
                raw_data[col] = raw_data[col].astype(float).interpolate(method='linear', limit_direction='both')
    return raw_data

# For Pytorch
def prepare_data(bentoml_logger, args, train_df, test_df, look_back=6,  batch_size=64):
    """
        Prepares data for a model, including normalization and splitting into features and labels.
        
        Parameters:
        - train_df (pd.DataFrame): Training data.
        - test_df (pd.DataFrame): Testing data.
        - target_column (str): Name of the target column.
        - look_back (int): Number of past observations to use as features.

        Returns:
        - data_components (dict): Dictionary containing training and testing data, scaler object, and model parameters.
    """
    n_out = 1 # defines the steps-ahead prediction 
        
    data_components = {}
    bentoml_logger.info(f"Preparing dataset for PyTorch model")
    scaled_data_train, scaled_data_test, scaler_obj = scale_data(train_df, test_df, scaler="MinMax")
    if args.model_type == 'ARIMA':
        bentoml_logger.info(f"Preparing dataset for ARIMA model")
        data_components = prep_data_components(X_train=scaled_data_train, X_test=scaled_data_test, scaler_obj=scaler_obj,
                                                   model_params=args.model_parameters["arima_model_parameters"])
        
    else:
        supervised_train_data = ts_supervised_structure(scaled_data_train, n_in=look_back, n_out=n_out)
        supervised_test_data = ts_supervised_structure(scaled_data_test, n_in=look_back, n_out=n_out)

        X_train_np = supervised_train_data.iloc[:, :-args.num_variables].values  # All features except the last two columns
        y_train_np = supervised_train_data.iloc[:, -args.num_variables:].values  # Only the last two columns
        X_test_np = supervised_test_data.iloc[:, :-args.num_variables].values
        y_test_np = supervised_test_data.iloc[:, -args.num_variables:].values

        X_train = X_train_np.reshape(X_train_np.shape[0], look_back, args.num_variables)
        X_test = X_test_np.reshape(X_test_np.shape[0], look_back, args.num_variables)

        if args.model_type == 'PYTORCH':
            # Convert to PyTorch tensors
            X_train = torch.from_numpy(X_train).float()
            y_train = torch.from_numpy(y_train_np).float()
            X_test = torch.from_numpy(X_test).float()
            y_test = torch.from_numpy(y_test_np).float()
                
            # Create PyTorch datasets
            train_dataset = TimeSeriesDataset(X_train, y_train)
            test_dataset = TimeSeriesDataset(X_test, y_test)
            bentoml_logger.info("PREPARE DATA self.data_components['model_parameters']: ", args.model_parameters)
            data_components = prep_data_components(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, 
                                                scaler_obj=scaler_obj, model_params=args.model_parameters["pytorch_model_parameters"],
                                                train_dataset=train_dataset, test_dataset=test_dataset, batch_size=batch_size)
        elif args.model_type == 'XGB':
            supervised_train_data = ts_supervised_structure(train_df, n_in=look_back, n_out=1)
            supervised_test_data = ts_supervised_structure(test_df, n_in=look_back, n_out=1)
            X_train = supervised_train_data.iloc[:, :-1]
            y_train = supervised_train_data.iloc[:, -1]
            X_test = supervised_test_data.iloc[:, :-1]
            y_test = supervised_test_data.iloc[:, -1]
            X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized, scaler_obj = normalize_data(X_train, X_test, y_train, y_test)
            data_components = prep_data_components(X_train=X_train_normalized, X_test=X_test_normalized, y_train=y_train_normalized, y_test=y_test, 
                                                    scaler_obj=scaler_obj, model_params=args.model_parameters["xgboost_model_parameters"])  
            

    return data_components


def create_nanny_ml_df(data_components=0, test_df=0, model = None, args=None, phase='test'):
                            # automatically update to train or testing data as required
                            X = data_components[f"X_{phase}"]# change by test_data as it contains datetime index
                            y = data_components[f"y_{phase}"]
                            if args.model_type == 'XGB':
                                y_pred = model.predict(X)
                                data = {'y_true': y.astype(np.float64), 'y_pred': y_pred.astype(np.float64)}
                                data = pd.DataFrame(data, dtype='object')
                                data.reset_index(drop=True, inplace=True)
                            elif args.model_type == 'ARIMA':
                                y_pred = data_components["y_pred"]
                                data = {'y_true': y.squeeze().astype(np.float64), 'y_pred': y_pred.squeeze().astype(np.float64)}
                                data = pd.DataFrame(data, dtype='object')
                                data.reset_index(drop=True, inplace=True)
                            return data
