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

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # to be able to call `dataset/`
from intelligence.adapters import dataclay_client
from tai.monitoring import *
from tai.model_explainability import *
from processing.process import *


def execute_pytorch(results, bentoml_logger, ModelMetricsDataClay, args, mlflow, bentoml, datasplit, data_components):
    with dataclay_client(args), mlflow.start_run():
        bentoml_logger.info("Training Pytorch model")
        bentoml_logger.info("Running model training %s",
                            "inside dataclay" if args.dataclay else "locally in device")

        model = ModelMetricsDataClay(data_components)
        if args.dataclay:
            model.make_persistent()
            bentoml_logger.info("Persistent PyTorch model object created!")

        metrics_dict, model, step_epoch, step_loss = model.train_pytorch(args.device)

        mlflow.log_param("model_params", data_components['model_parameters'])
        for epoch, loss in zip(step_epoch, step_loss):
            mlflow.log_metric("loss", float(loss), step=epoch)

        # Log metrics for each variable dynamically
        for name, metric in metrics_dict.items():
            prefix = f"Metrics_{name}"
            mlflow.log_metric(f"{prefix}/MAE", metric['mae'])
            mlflow.log_metric(f"{prefix}/MSE", metric['mse'])
            mlflow.log_metric(f"{prefix}/RMSE", metric['rmse'])
            mlflow.log_metric(f"{prefix}/MAPE", metric['mape'])
            mlflow.log_metric(f"{prefix}/SMAPE", metric['smape'])

        custom_objects = {
            "scaler_obj": data_components['scaler_obj'],
            "test_sample_size": len(data_components['X_test']),
            "model_metrics": metrics_dict,
        }

        bento_model = bentoml.picklable_model.save_model(
            'metrics_utilization_model_lstm',
            model,
            custom_objects=custom_objects,
        )

        mlflow.log_params(datasplit)
        mlflow.set_tag("bentoml.model", bento_model.tag)

        save_shap_explanations_pytorch(model, data_components, args, mlflow, bentoml_logger, args.model_type)

    mlflow.end_run()

    results["Trained Model"] = str(bento_model.tag)
    results["Dataset Splits"] = datasplit
    results["Model Metrics"] = metrics_dict

    bentoml_logger.info(f"Model saved: {bento_model}")
    bentoml_logger.info(f"\nSaved Model in BentoML: {bento_model}")
    bentoml_logger.info(f"Trained model\n: {model}")

    return save_results(bentoml_logger, results, datasplit, bento_model, metrics_dict)