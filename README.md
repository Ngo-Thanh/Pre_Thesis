# Heart Disease Classification Prototype

This repository contains a proof-of-concept LightGBM model for predicting the
heart disease status of a patient based on data such as blood pressure,
cholesterol level and lifestyle habits. The model and Streamlit app are
research prototypes and **not** intended for clinical use.

## Training the model

```bash
python train_model.py
```
This will generate a file named `lgbm_model.pkl` containing the preprocessing
pipeline and trained LightGBM classifier.

## Running the demo app

```bash
streamlit run doctor_app.py
```
The application prompts for patient information and outputs the predicted heart
disease status. The prediction is for educational purposes only and should not
be used as medical advice.
