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

import shap
import matplotlib.pyplot as plt
import matplotlib
import torch

from analytics.lstm_model import LSTMModel
import numpy as np
matplotlib.use('Agg')  # Use non-GUI backend

def save_shap_explanations(model, data_components, args, mlflow, bentoml_logger, model_type="default"):
    """
    Calculate and save SHAP explanations for the model using MLflow.

    Parameters:
    - model: The trained model for which SHAP explanations are calculated.
    - data_components: Data used for SHAP explanation.
    - args: Additional arguments passed to the SHAP calculation function.
    - mlflow: MLflow module for logging artifacts and explanations.
    - bentoml_logger: Logger for tracking exceptions or info.
    - model_type: The type of the model (e.g., 'default', 'arima').
    """
    try:
        shap_explainer = shap.Explainer(model.predict, data_components['X_test'][:args.shap_samples])
        shap_values = shap_explainer(data_components['X_test'][:args.shap_samples])
        
        plots = [
            (shap.plots.waterfall, shap_values[1], f"model_explanations_shap/summary_waterfall_plot_{model_type}.png"),
            (shap.plots.bar, shap_values, f"model_explanations_shap/summary_bar_plot_{model_type}.png"),
            (shap.plots.heatmap, shap_values, f"model_explanations_shap/summary_beeswarm_plot_{model_type}.png")
        ]
        for plot_func, shap_data, file_name in plots:
            fig, _ = plt.subplots(figsize=(10, 8))  
            plot_func(shap_data, show=False)
            mlflow.log_figure(figure=fig, artifact_file=file_name)
        mlflow.shap.log_explainer(explainer=shap_explainer,artifact_path="model_explanations_shap",serialize_model_using_mlflow=False)
        bentoml_logger.info("Saving model explainability for XGBoost model")
    except Exception as e:
                    bentoml_logger.exception(e)


def save_shap_explanations_pytorch(model, data_components, args, mlflow, bentoml_logger, model_type="default"):
    # No-op. Upstream PyTorch SHAP path was never wired up — explainability
    # for PyTorch is a phase-2 (or later) item; see intelligence-utility-plan.v2.md §3.3.
    return None