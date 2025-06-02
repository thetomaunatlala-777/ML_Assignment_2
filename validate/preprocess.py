import pandas as pd

# Just converted the cells from Jupyter notebook into functions, I left the comments to guide you, you can remove them upon submission


def load_data():
    """Load training features and labels from text files."""
    # Load training features
    X_train = pd.read_csv("traindata.txt", header=None, sep=',')

    # Load training labels
    y_train = pd.read_csv("trainlabels.txt", header=None, names=['label'])

    print("Training features shape:", X_train.shape)
    print("Training labels shape:", y_train.shape)

    return X_train, y_train


def combine_data(X_train, y_train):
    """Combine features and labels into a single dataframe."""
    df = X_train.copy()
    df["label"] = y_train.values.ravel()
    return df


def handle_missing_values(df):
    """Fill missing values with median for specified columns."""
    columns_with_missing = [1, 5, 6, 7, 8, 14, 15, 17, 27, 34, 40]

    for col in columns_with_missing:
        df[col] = df[col].fillna(df[col].median())

    print("Missing values after filling:", df.isnull().sum().sum())
    return df


def remove_text_column(df):
    """Remove column 46 (text column with low predictive power)."""
    # Note: We already analyzed that column 46 has low predictive power
    df = df.drop(columns=df.columns[46])
    return df


def select_top_features(df, n_features=20):
    """Select top n features based on correlation with labels."""
    features = df.iloc[:, :-1]
    labels = df.iloc[:, -1].astype(int)

    # Calculate correlations
    correlations = features.corrwith(labels).abs().sort_values(ascending=False)

    # Select top features
    top_features = correlations.head(n_features).index
    X_selected = features[top_features]

    # Combine selected features with labels
    df_selected = pd.concat([X_selected, labels], axis=1)

    return df_selected


def preprocess_data():
    """Main preprocessing pipeline."""
    # Step 1: Load data
    X_train, y_train = load_data()

    # Step 2: Combine features and labels
    df = combine_data(X_train, y_train)

    # Step 3: Handle missing values
    df = handle_missing_values(df)

    # Step 4: Remove text column with low predictive power
    df = remove_text_column(df)

    # Step 5: Select top features based on correlation
    df_selected = select_top_features(df, n_features=20)

    print(f"Original shape: {df.shape}")
    print(f"Final shape after preprocessing: {df_selected.shape}")

    return df_selected


def save_preprocessed_data(df_selected):
    """Save preprocessed data to CSV file."""
    df_selected.to_csv("preprocessed_data.txt", index=False)
    print("Preprocessed data saved to 'preprocessed_data.txt'")


if __name__ == "__main__":
    # Run preprocessing
    preprocessed_df = preprocess_data()

    # Save the preprocessed data
    save_preprocessed_data(preprocessed_df)
