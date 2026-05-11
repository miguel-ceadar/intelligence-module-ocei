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
from tai.monitoring import create_drift_detector
from processing.process import *


def execute_arima(results, bentoml_logger, ModelMetricsDataClay, args, mlflow, bentoml, datasplit, data_components, train_df, test_df):
    with dataclay_client(args), mlflow.start_run():
        bentoml_logger.info("Training ARIMA model")
        bentoml_logger.info("Running model training %s",
                            "inside dataclay" if args.dataclay else "locally in device")

        model = ModelMetricsDataClay(data_components)
        if args.dataclay:
            model.make_persistent()
            bentoml_logger.info("Persistent ARIMA model object created!")

        mlflow.log_param("model_params", data_components['model_parameters'])
        metrics, model, history, y_test, y_pred = model.train_arima()
        data_components["y_test"] = y_test
        data_components["y_pred"] = y_pred
        # Handle drift detection and save model
        reference_df = create_nanny_ml_df(data_components, test_df, model, args, "test")
        drift_detector_obj = create_drift_detector(["y_pred"], args.steps_back, reference_df)

        bento_model = bentoml.picklable_model.save_model('metrics_utilization_model_arima', model,
                                                         custom_objects={"scaler_obj": data_components['scaler_obj'],
                                                                         "historical_data": history,
                                                                         "drift_detector_obj": drift_detector_obj,
                                                                         "model_metrics": metrics,
                                                                         "test_sample_size": len(data_components['X_test'])},
                                                         signatures={"predict": {"batchable": True}})

        mlflow.log_metrics(metrics)
        mlflow.log_params(datasplit)
        mlflow.set_tag("bentoml.model", bento_model.tag)
    mlflow.end_run()

    return save_results(bentoml_logger, results, datasplit, bento_model, metrics)
