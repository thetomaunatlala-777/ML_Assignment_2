import numpy as np
import pandas as pd

N_CLASSES = 10


def calculate_priors(train_labels):
    """Calculate prior probabilities for each class."""
    priors = np.zeros(N_CLASSES)
    unique_labels, counts = np.unique(train_labels, return_counts=True)
    total_samples = len(train_labels)

    for label, count in zip(unique_labels, counts):
        priors[int(label)] = count / total_samples

    return priors


def calculate_means(train_data, train_labels):
    """Calculate mean for each feature in each class."""
    n_features = train_data.shape[1]
    means = np.zeros((n_features, N_CLASSES))

    for class_label in range(N_CLASSES):
        class_filter = (train_labels == class_label)
        class_data = train_data[class_filter]

        if len(class_data) > 0:
            for feature_idx in range(n_features):
                feature_values = class_data[:, feature_idx]
                means[feature_idx, class_label] = np.mean(feature_values)

    return means


def calculate_variances(train_data, train_labels, means):
    """Calculate variance for each feature in each class."""
    n_features = train_data.shape[1]
    variances = np.zeros((n_features, N_CLASSES))

    for class_label in range(N_CLASSES):
        class_mask = (train_labels == class_label)
        class_data = train_data[class_mask]

        if len(class_data) > 0:
            for feature_idx in range(n_features):
                feature_values = class_data[:, feature_idx]
                variance = np.var(feature_values)
                variances[feature_idx, class_label] = variance + 1e-9

    return variances


def train_and_save_model():
    # Load and preprocess training data
    train_df = pd.read_csv("preprocessed_data.txt")

    train_data = train_df.iloc[:, :-1].values
    train_labels = train_df.iloc[:, -1].values

    # Get selected feature indices
    selected_features = train_df.columns[:-1].tolist()
    selected_feature_indices = [int(col) for col in selected_features]

    # Train model
    priors = calculate_priors(train_labels)
    means = calculate_means(train_data, train_labels)
    variances = calculate_variances(train_data, train_labels, means)

    # Save model parameters using numpy
    np.save('priors.npy', priors)
    np.save('means.npy', means)
    np.save('variances.npy', variances)
    np.save('selected_features.npy', np.array(selected_feature_indices))

    print("Model parameters saved")


if __name__ == "__main__":
    train_and_save_model()
