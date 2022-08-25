import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import argparse

parser = argparse.ArgumentParser(description='NS-ANN prediction script')
parser.add_argument('--input', help='path to the input file')
parser.add_argument('--outpur', help='path to the output file')
parser.add_argument('--model', help='path to the pretrained model')
args = parser.parse_args()

# Import EOS data
data = pd.read_csv(args.input)

# Load the model
model = tf.keras.models.load_model(args.model)

# Scale the data
X_scaler = joblib.load('./utils/X_scaler.pkl')
y_scaler = joblib.load('./utils/y_scaler.pkl')
X = X_scaler.transform(data.iloc[:, 0:7].values)

# Predict NS properties
y = model.predict(X)
y = y_scaler.inverse_transform(y)

# Write to output
y = pd.DataFrame(y, columns=["NS mass", "Rmax", "R14", "Lambda10", "Lambda14", "Lambda18"])
y.to_csv(args.output, index=False)