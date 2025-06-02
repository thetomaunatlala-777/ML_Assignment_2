#  Import any Python standard libraries you wish   #
# - I.e. libraries that do not require pip install with fresh
#   install of Python #

##################################
# ALLOWED NON-STANDARD LIBRARIES #
##################################
# Un-comment out the ones you use
import numpy as np
import pandas as pd
# import sklearn
# import torch
# import matplotlib
# import seaborn as sns
##################################

N_CLASSES = 10


def gaussian_pdf(x, mean, variance):
    """Calculate Gaussian probability density function."""
    coefficient = 1.0 / np.sqrt(2 * np.pi * variance)
    exponent = -((x - mean) ** 2) / (2 * variance)
    return coefficient * np.exp(exponent)


def calculate_posterior(feature, class_label, priors, means, variances):
    """Calculate posterior probability for a given class and feature vector."""
    likelihood = 1.0
    for feature_idx in range(len(feature)):
        mean = means[feature_idx, class_label]
        var = variances[feature_idx, class_label]
        feature_likelihood = gaussian_pdf(feature[feature_idx], mean, var)
        likelihood *= feature_likelihood

    numerator = likelihood * priors[class_label]

    denominator = 0.0
    for c in range(N_CLASSES):
        class_likelihood = 1.0
        for f_idx in range(len(feature)):
            mean_c = means[f_idx, c]
            var_c = variances[f_idx, c]
            class_likelihood *= gaussian_pdf(feature[f_idx], mean_c, var_c)
        denominator += class_likelihood * priors[c]

    posterior = numerator / (denominator + 1e-10)
    return posterior


def predict_class(feature, priors, means, variances):
    """Predict class for a single feature vector."""
    best_class = 0
    best_probability = 0.0

    for class_label in range(N_CLASSES):
        posterior_prob = calculate_posterior(
            feature, class_label, priors, means, variances)
        if posterior_prob > best_probability:
            best_probability = posterior_prob
            best_class = class_label

    return best_class


def main():
    # Load test data
    test_data = pd.read_csv("testdata.txt", header=None)
    n_datapoints = test_data.shape[0]

    # Load pre-trained model parameters
    priors = np.load('priors.npy')
    means = np.load('means.npy')
    variances = np.load('variances.npy')
    selected_features = np.load('selected_features.npy').astype(int)

    # Apply preprocessing to test data
    test_df = test_data.copy()

    # Handle missing values
    columns_with_missing = [1, 5, 6, 7, 8, 14, 15, 17, 27, 34, 40]
    for col in columns_with_missing:
        if col < test_df.shape[1]:
            test_df[col] = test_df[col].fillna(test_df[col].median())

    # Select features using the saved indices
    test_features = test_df.iloc[:, selected_features].values

    # Make predictions
    predictions = []
    for i in range(n_datapoints):
        pred = predict_class(test_features[i], priors, means, variances)
        predictions.append(pred)

    # Save predictions
    infer_labels = pd.DataFrame(predictions)
    infer_labels.to_csv("predlabels.txt", index=False, header=False)


if __name__ == "__main__":
    main()
