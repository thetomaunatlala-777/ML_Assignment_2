import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
X = pd.read_csv("traindata.txt", header=None, sep=',')
y = pd.read_csv("trainlabels.txt", header=None, names=['label'])

# Combine features and labels
df = X.copy()
df["label"] = y.values.ravel()

# Impute missing values with median
columns_with_missing = [1, 5, 6, 7, 8, 14, 15, 17, 27, 34, 40]
for col in columns_with_missing:
    df[col] = df[col].fillna(df[col].median())

# Drop random words column if exists
if df.shape[1] > 46:
    df = df.drop(columns=df.columns[46])

# Separate features and labels
features = df.iloc[:, :-1]
labels = df.iloc[:, -1].astype(int)

# Select best features by correlation
correlations = features.corrwith(labels).abs().sort_values(ascending=False)
top_features = correlations.head(10).index
X_selected = features[top_features]

np.savetxt("top_features.txt", top_features, fmt='%d')


# data split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, labels, test_size=0.2, random_state=42, stratify=labels
)

# scaling the features then using scaled versions going forward
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# we save our parameters into txt files
np.savetxt("scaler_mean.txt", scaler.mean_)
np.savetxt("scaler_scale.txt", scaler.scale_)

# logistic regression model on training data
model = LogisticRegression(
    solver='lbfgs',     
    penalty='l2',       
    C=0.7,              
    max_iter=5000,
    
)
# here we apply the model to training data split
model.fit(X_train_scaled, y_train)

# Save model weights and intercept
np.savetxt("model_weights.txt", model.coef_)
np.savetxt("model_intercept.txt", model.intercept_)

# this is the prediction on the test split we have
y_pred = model.predict(X_test_scaled)


from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred, average='weighted')

accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy:.4f}")
print(f"Weighted F1 Score: {f1:.4f}")
