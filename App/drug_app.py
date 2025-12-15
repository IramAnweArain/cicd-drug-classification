import skops.io as sio
import gradio as gr
import os

# Define the path to the model
model_path = "../Model/drug_pipeline.skops"

# --- SECURITY FIX START ---
# skops 0.10+ requires explicit trust. We inspect the file first.
# This gets the list of types inside the model file so we can trust them.
untrusted_types = sio.get_untrusted_types(file=model_path)
pipe = sio.load(model_path, trusted=untrusted_types)
# --- SECURITY FIX END ---

def predict_drug(age, sex, bp, cholesterol, na_to_k):
    # Format input exactly as the model expects (DataFrame)
    import pandas as pd
    features = pd.DataFrame([[age, sex, bp, cholesterol, na_to_k]], 
                            columns=["Age", "Sex", "BP", "Cholesterol", "Na_to_K"])
    prediction = pipe.predict(features)[0]
    return prediction

# Create Gradio Interface
inputs = [
    gr.Slider(15, 74, step=1, label="Age"),
    gr.Radio(["M", "F"], label="Sex"),
    gr.Radio(["HIGH", "LOW", "NORMAL"], label="Blood Pressure"),
    gr.Radio(["HIGH", "NORMAL"], label="Cholesterol"),
    gr.Slider(0, 30, label="Na_to_K Ratio"),
]

outputs = gr.Label(label="Predicted Drug")

examples = [
    [30, "M", "HIGH", "NORMAL", 15.4],
    [50, "F", "NORMAL", "HIGH", 10.1]
]

title = "Drug Classification App"
description = "Enter patient details to predict the correct drug."

gr.Interface(fn=predict_drug, inputs=inputs, outputs=outputs, examples=examples, title=title, description=description).launch()