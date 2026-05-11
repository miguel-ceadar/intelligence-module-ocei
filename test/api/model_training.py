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

# Test POST request API to perform model training

import requests

url = 'http://localhost:3000/train'

headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}

test_cases = {
    'test_model_training_locally_xgb' : {
                                        'model_name': 'METRICS',
                                        'model_type': 'XGB',
                                        'test_size': 0.2,
                                        'dataset_name': 'cpu_sample_dataset_orangepi.csv',
                                        'steps_back': 12,
                                        "num_variables": 1,
                                        'max_models_count': 5,
                                        'max_mlruns_count': 10,
                                        'model_parameters': {
                                            'xgboost_model_parameters': {
                                                'n_estimators': 1000,
                                                'max_depth': 7,
                                                'eta': 0.1,
                                                'subsample': 0.7,
                                                'colsample_bytree': 0.8,
                                                'alpha': 0,
                                            },
                                            },
                                        },
    'test_model_training_locally_arima': {
                                        'model_name': 'METRICS',
                                        'model_type': 'ARIMA',
                                        'test_size': 0.2,
                                        'dataset_name': 'cpu_sample_dataset_orangepi.csv',
                                        "num_variables": 1,
                                        'steps_back': 12,
                                        'max_models_count': 5,
                                        'max_mlruns_count': 10,
                                        'dataclay': False,
                                        'model_parameters': {
                                            'arima_model_parameters': {
                                                'p': 5,
                                                'd': 1,
                                                'q': 0,
                                            },
                                        },
                                    },
   'test_model_training_locally_lstm': {
                                        "model_name": "METRICS",
                                        "model_type": "PYTORCH",
                                        "test_size": 0.2,
                                        "dataset_name": "node_3_utilisation_sample_dataset.csv",
                                        "num_variables": 2,
                                        "steps_back": 6,
                                        "batch_size": 64,
                                        "max_models_count": 5,
                                        "max_mlruns_count": 10,
                                        "shap_samples": 100,
                                        "dataclay": False,
                                        "dataclay_host": "127.0.0.1",
                                        "dataclay_hostname": "testuser",
                                        "dataclay_password": "s3cret",
                                        "dataclay_dataset": "testdata",
                                        "model_parameters": {
                                            "pytorch_model_parameters": {
                                            "distill": False,
                                            "hidden_size": 64,
                                            "input_size": 2,
                                            "num_epochs": 50,
                                            "output_size": 2,
                                            "quantize": False
                                            }
                                        },
                                        "device": "cpu",
                                        "federated": {
                                                "use_federated": False,
                                                "federated_params": {
                                                    "run_config": {},
                                                    "use_stream": True
                                                }
                                        }
                                    }
}


def launch_test_training():
    for test in test_cases:
        print('-'*50)
        print(f"Running test: {test}")
        response = requests.post(url=url, headers=headers, json=test_cases[test])
        if response.status_code == 200:
            print('Test passed')
            print(response.json())

        else:
            print(response)
            print('Test failed')
        print('-'*50)

if __name__ == "__main__":
    launch_test_training()
