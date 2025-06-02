#classifyall.py

import pandas as pd
import numpy as np

# load saved parameters
top_features = np.loadtxt("top_features.txt", dtype=int)
scaler_mean = np.loadtxt("scaler_mean.txt")
scaler_scale = np.loadtxt("scaler_scale.txt")
model_weights = np.loadtxt("model_weights.txt")
model_intercept = np.loadtxt("model_intercept.txt")

# load test data
X_test = pd.read_csv("testdata.txt", header=None, sep=',')

# impute with median
for col in X_test.columns:
    if X_test[col].isnull().any():
        X_test[col] = X_test[col].fillna(X_test[col].median())



# drop random words column if it is in testdata
if X_test.shape[1] > 46:
    X_test = X_test.drop(columns=X_test.columns[46])

# select top features
X_selected = X_test.iloc[:, top_features]

# scale features
X_scaled = (X_selected - scaler_mean) / scaler_scale

# predict using saved model weights
logits = np.dot(X_scaled, model_weights.T) + model_intercept

# softmax function for multiclass prediction
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # stability
    return exp_z / exp_z.sum(axis=1, keepdims=True)

probabilities = softmax(logits)
predictions = np.argmax(probabilities, axis=1)

# save predictions
pd.DataFrame(predictions).to_csv("predlabels.txt", index=False, header=False)

print("Predictions saved to predlabels.txt")
