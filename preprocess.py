import numpy as np
from helpers import *


def extract_headers(csv_file_path):
    """
    This function extracts the headers (feature names) from the given CSV file.
    
    Args:
        csv_file_path (str): Path to the CSV file from which to extract headers.
        
    Returns:
        list: A list of headers (feature names), excluding the 'Id' column.
    """
    with open(csv_file_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # Read the first row as headers
    return headers[1:]  # Exclude the first column header as it is 'Id'


def replace_dont_know_refused_all(data):
    """
    Replaces values of 7 and 9 with NaN across all features in the dataset.

    Args:
        data (np.array): The dataset with features as columns.

    Returns:
        np.array: The dataset with 7 and 9 replaced by NaN in all columns.
    """
    # Replace 7 and 9 with NaN in the entire dataset
    data_cleaned = np.where(
    (data == 7) | (data == 77) | (data == 777) | (data == 7777) | (data == 77777) | (data == 777777) |
    (data == 9) | (data == 99) | (data == 999) | (data == 9999) | (data == 99999) | (data == 999999),
    np.nan,
    data
)
    
    return data_cleaned


def identify_feature_types(data, categorical_threshold=15):
    """
    Identifies the types of features in a dataset and returns indices for binary, continuous, or categorical features.

    Args:
        data (np.array): The dataset as a numpy array with features as columns.
        headers (list): List of feature names corresponding to data columns.
        categorical_threshold (int): The maximum unique value count to classify a feature as categorical.

    Returns:
        dict: A dictionary with lists of indices for binary, continuous, and categorical features.
        dict: A dictionary with counts of each feature type.
    """
    # Initialize lists for indices of each feature type
    binary_indices = []
    continuous_indices = []
    categorical_indices = []

    # Analyze each feature column-wise
    for col_idx in range(data.shape[1]):
        column_data = data[:, col_idx]
        
        # Remove missing or NaN values for accurate counting
        column_data = column_data[~np.isnan(column_data)]
        
        # Get unique values
        unique_values = np.unique(column_data)
        unique_count = len(unique_values)

        # Classify features based on unique values
        if unique_count == 2:
            binary_indices.append(col_idx)
        elif unique_count > categorical_threshold:
            continuous_indices.append(col_idx)
        else:
            categorical_indices.append(col_idx)

    # Prepare the output dictionaries
    feature_indices = {
        "binary_indices": binary_indices,
        "continuous_indices": continuous_indices,
        "categorical_indices": categorical_indices,
    }

    feature_counts = {
        "binary_count": len(binary_indices),
        "continuous_count": len(continuous_indices),
        "categorical_count": len(categorical_indices),
    }

    return feature_indices, feature_counts


def remove_columns_by_index(data, columns_to_remove):
    """
    Removes specified columns from the dataset based on column indices.

    Args:
        data (np.array): The dataset from which to remove columns.
        columns_to_remove (list): List of column indices to remove.

    Returns:
        np.array: The dataset with specified columns removed.
    """
    # Remove columns from data using indices
    data_filtered = np.delete(data, columns_to_remove, axis=1)
    
    print(f"Removed columns at indices: {columns_to_remove}")
    return data_filtered


def remove_missing_values(data, threshold=0.25):
    """
    Removes features (columns) from the dataset that have more than the specified
    percentage of missing values.

    Args:
        headers (list): The list of feature names.
        data (np.array): The dataset (numpy array) with features as columns.
        threshold (float): The maximum allowed percentage of missing values (default is 25%).

    Returns:
        tuple: A tuple containing:
            - valid_columns (list): The indices of the features that were kept.
            - filtered_data (np.array): The filtered dataset with only the valid columns.
    """
    # Find the percentage of missing values for each column
    missing_values = np.isnan(data).mean(axis=0)
    valid_columns = np.where(missing_values <= threshold)[0]
    
    # Filter out the columns with too many missing values
    filtered_data = data[:, valid_columns]
    return valid_columns, filtered_data


def replace_binary_values(data, binary_columns_indices):
    """
    Replaces all values of 1 with 0 and values of 2 with 1 in the specified binary feature columns.

    Args:
        data (np.array): The dataset containing binary features.
        binary_columns_indices (list): List of indices representing binary columns to transform.

    Returns:
        np.array: The dataset with transformed binary values.
    """
    for col in binary_columns_indices:
        # Apply the replacement only to columns specified as binary
        data[:, col] = np.where(data[:, col] == 1, 0, 
                                np.where(data[:, col] == 2, 1, data[:, col]))

    return data


def split_and_balance_data(x_train, y_train, test_size):
    """
    Balances x_train and y_train to have an equal number of -1 and 1 labels.
    
    Args:
        x_train: numpy array of shape (N, D), where N is the number of samples and D is the number of features.
        y_train: numpy array of shape (N,), where N is the number of samples and each entry is either 1 or -1.
        
    Returns:
        x_train_balanced: numpy array of balanced samples.
        y_train_balanced: numpy array of balanced labels.
    """
    # Find indices for each class
    indices_class_1 = np.where(y_train == 1)[0]
    indices_class_neg_1 = np.where(y_train == -1)[0]
    
    # Determine the smaller class size
    min_class_size = min(len(indices_class_1), len(indices_class_neg_1))
    
    # Randomly select indices to balance the classes
    balanced_indices_class_1 = np.random.choice(indices_class_1, min_class_size, replace=False)
    balanced_indices_class_neg_1 = np.random.choice(indices_class_neg_1, min_class_size, replace=False)
    
    # Combine indices and shuffle
    balanced_indices = np.concatenate([balanced_indices_class_1, balanced_indices_class_neg_1])
    np.random.shuffle(balanced_indices)
    
    # Filter x_train and y_train based on balanced indices
    x_train_balanced = x_train[balanced_indices]
    y_train_balanced = y_train[balanced_indices]
    
    # Mélanger les indices de manière aléatoire
    indices = np.arange(x_train_balanced.shape[0])
    np.random.shuffle(indices)

    # Calculer le point de séparation pour un ratio basé sur test_size
    split_index = int(test_size * x_train_balanced.shape[0])

    # Séparer les indices pour l'entraînement et la validation
    train_indices = indices[:split_index]
    val_indices = indices[split_index:]

    # Créer les ensembles d'entraînement et de validation
    x_train, y_train = x_train_balanced[train_indices], y_train_balanced[train_indices]
    x_val, y_val = x_train_balanced[val_indices], y_train_balanced[val_indices]
    
    return x_train, x_val, y_train, y_val


def standardize_continuous_features(data, continuous_indices):
    """
    Standardizes only continuous features in the dataset.

    Args:
        data (np.array): The dataset with features as columns.
        continuous_indices (list): List of indices representing continuous features.

    Returns:
        np.array: The dataset with continuous features standardized.
    """
    # Copy data to avoid modifying the original dataset
    data_standardized = data.copy()

    for col_idx in continuous_indices:
        column_data = data[:, col_idx]
        
        # Calculate mean and std, ignoring NaN values
        mean = np.nanmean(column_data)
        std = np.nanstd(column_data)
        
        # Standardize only if std is non-zero to avoid division by zero
        if std > 0:
            data_standardized[:, col_idx] = (column_data - mean) / std

    return data_standardized


def determine_one_hot_categories(data, categorical_columns):
    """
    Determines unique categories across all categorical features for consistent one-hot encoding.
    
    Args:
        data (np.array): Dataset to analyze.
        categorical_columns (list): Indices of categorical columns.

    Returns:
        dict: Dictionary where keys are column indices, and values are lists of unique categories.
    """
    categories_per_column = {}
    for col_idx in categorical_columns:
        unique_categories = np.unique(data[~np.isnan(data[:, col_idx]), col_idx])
        categories_per_column[col_idx] = unique_categories
    return categories_per_column


def one_hot_encode(data, categorical_columns):
    """
    Applies one-hot encoding to specified categorical columns by transforming each unique category 
    into a binary column.
    
    Args:
        data (np.array): Dataset to encode.
        categorical_columns (list): Indices of categorical columns.

    Returns:
        np.array: Dataset with one-hot encoded features.
        list: List of binary (one-hot encoded) column indices.
    """
    updated_data = []
    binary_columns_indices = []
    current_index = 0

    for col_idx in range(data.shape[1]):
        if col_idx in categorical_columns:
            # Determine unique categories in the column (excluding NaN values)
            unique_categories = np.unique(data[~np.isnan(data[:, col_idx]), col_idx])
            
            # Create a binary column for each unique category
            for category in unique_categories:
                binary_column = (data[:, col_idx] == category).astype(int)
                updated_data.append(binary_column)
                
                # Track the binary column index
                binary_columns_indices.append(current_index)
                current_index += 1
        else:
            updated_data.append(data[:, col_idx])
            current_index += 1

    # Stack the updated data columns horizontally
    updated_data = np.column_stack(updated_data)
    return updated_data, binary_columns_indices


def mean_imputation(data, binary_columns_indices):
    """
    Imputes missing values in the dataset by replacing them with the mean of each column.
    For binary columns, the mean is rounded to 0 or 1.

    Args:
        data (np.array): The dataset with missing values (numpy array with NaN for missing values).
        binary_columns_indices (list): List of indices representing binary columns.

    Returns:
        np.array: The dataset with missing values replaced by column means or rounded means for binary features.
    """
    # Compute the mean for each column, ignoring NaNs
    col_means = np.nanmean(data, axis=0)
    
    # Adjust means for binary columns to ensure they are either 0 or 1
    for col in binary_columns_indices:
        col_means[col] = round(col_means[col])

    # Find indices where NaN values are present in the data
    inds = np.where(np.isnan(data))
    
    # Replace NaNs with the computed means
    data[inds] = np.take(col_means, inds[1])
    
    return data


def variance_thresholding(data, threshold=0.01):
    """
    Removes features that have variance below a specified threshold.

    Args:
        data (np.array): The dataset (numpy array) with features as columns.
        threshold (float): The minimum variance threshold (default is 1%).

    Returns:
        tuple: A tuple containing:
            - valid_columns (list): The indices of the features that were kept.
            - filtered_data (np.array): The filtered dataset with only the valid columns.
    """
    # Compute variance for each column
    variances = np.nanvar(data, axis=0)
    
    # Identify columns that meet the variance threshold
    valid_columns = np.where(variances >= threshold)[0]
    removed_columns = np.where(variances < threshold)[0]

    # Filter out the columns with variance below the threshold
    filtered_data = data[:, valid_columns]

    print(f"Variance thresholding: {len(removed_columns)} features removed with variance below {threshold}")
    
    return valid_columns, filtered_data


def correlation_analysis(x_data, y_data, low_threshold=0.05, high_threshold=0.9):
    """
    Removes features that have low correlation with the outcome variable (y_data),
    or that are highly correlated with other features.

    Args:
        headers (list): The list of feature names.
        x_data (np.array): The feature dataset (numpy array) with features as columns.
        y_data (np.array): The outcome variable (labels) to correlate with.
        low_threshold (float): Minimum Pearson correlation threshold (default is 0.05).
        high_threshold (float): Maximum Pearson correlation threshold (default is 0.9).

    Returns:
        tuple: A tuple containing:
            - valid_columns (list): The indices of the features that were kept.
            - filtered_data (np.array): The filtered dataset with only the valid columns.
    """
    valid_columns = []

    # Ensure feature and outcome have the same non-NaN indices
    for i in range(x_data.shape[1]):
        feature = x_data[:, i]

        # Create a mask where both feature and y_data are non-NaN
        non_nan_mask = ~np.isnan(feature) & ~np.isnan(y_data)

        # Apply mask to both feature and y_data
        feature_clean = feature[non_nan_mask]
        y_data_clean = y_data[non_nan_mask]

        # Compute correlation if both feature_clean and y_data_clean have enough data
        if len(feature_clean) > 1 and len(y_data_clean) > 1:
            correlation = np.corrcoef(feature_clean, y_data_clean)[0, 1]

            # Check if the correlation is within the desired range
            if abs(correlation) >= low_threshold and abs(correlation) <= high_threshold:
                valid_columns.append(i)

    # Filter valid columns
    filtered_data = x_data[:, valid_columns]
    return valid_columns, filtered_data


def statistical_test_analysis(x_data, y_data, continuous_indices, p_value_threshold=0.05):
    """
    Retains features that have a statistically significant association with the outcome variable (y_data),
    using an F-test for continuous features and a Chi-Square test for categorical features.

    Args:
        x_data (np.array): The feature dataset (numpy array) with features as columns.
        y_data (np.array): The outcome variable (labels) to test associations with.
        continuous_indices (list): List of indices for continuous features in x_data.
        p_value_threshold (float): The significance level threshold for retaining features (default is 0.05).

    Returns:
        tuple: A tuple containing:
            - valid_columns (list): The indices of the features that were kept.
            - filtered_data (np.array): The filtered dataset with only the valid columns.
    """
    valid_columns = []
    removed_columns = []

    # Iterate over each feature
    for i in range(x_data.shape[1]):
        feature = x_data[:, i]
        
        # Create a mask where both feature and y_data are non-NaN
        non_nan_mask = ~np.isnan(feature) & ~np.isnan(y_data)
        feature_clean = feature[non_nan_mask]
        y_data_clean = y_data[non_nan_mask]

        # Determine if the feature is continuous based on index
        if i in continuous_indices:
            # Perform F-test for continuous features
            classes = [feature_clean[y_data_clean == label] for label in np.unique(y_data_clean)]
            overall_var = np.var(feature_clean)
            between_group_var = np.var([np.mean(cls) for cls in classes])
            f_statistic = between_group_var / overall_var if overall_var != 0 else 0
            
            # Calculate p-value approximation based on F-statistic
            p_value = 1 - (1 / (1 + f_statistic))  # Simplified approximation

        else:
            # Perform Chi-Square test for categorical features
            unique_y = np.unique(y_data_clean)
            unique_vals = np.unique(feature_clean)
            contingency_table = np.array([
                [(feature_clean == val).sum() & (y_data_clean == label).sum() for label in unique_y] for val in unique_vals
            ])
            chi_square_statistic = np.sum((contingency_table - np.mean(contingency_table))**2 / np.mean(contingency_table))
            p_value = 1 - (1 / (1 + chi_square_statistic))  # Simplified approximation

        # Determine if the feature is statistically significant
        if p_value < p_value_threshold:
            valid_columns.append(i)
        else:
            removed_columns.append(i)

    # Filter valid columns
    filtered_data = x_data[:, valid_columns]
    return valid_columns, filtered_data


def Preprocess_Data(
    file_path, x_train, y_train, x_test, test_size=0.2, balance_method="oversample",
    missing_val_threshold=0.25, variance_threshold=0.01,
    low_corr_threshold=0.05, high_corr_threshold=0.9, p_value_threshold=0.05
):
    """
    Preprocesses the dataset by sequentially applying data cleaning, feature selection, 
    standardization, and balancing steps.
    
    Args:
        file_path (str): Path to the dataset folder.
        x_train (np.array): Training feature dataset.
        y_train (np.array): Target variable for training.
        test_size (float): Proportion of data for the test set.
        balance_method (str): Method for balancing the training set ("undersample" or "oversample").
        missing_val_threshold (float): Threshold for removing columns with too many missing values.
        variance_threshold (float): Minimum variance threshold to retain features.
        low_corr_threshold (float): Minimum correlation threshold with target variable.
        high_corr_threshold (float): Maximum correlation threshold for feature redundancy.
        p_value_threshold (float): Threshold for feature selection based on statistical significance.
    
    Returns:
        np.array: Preprocessed and balanced training and test sets, including final `x_train`, 
        `x_test`, `y_train_balanced`, and `y_test`.
    """
    # Extract headers for reference
    x_train_headers = extract_headers(file_path + "x_train.csv")
    
    # Step 1: Replace "don't know" or "refused" values with NaN across all features
    x_train = replace_dont_know_refused_all(x_train)
    x_test = replace_dont_know_refused_all(x_test)
    
    
    '''# Step 2: Remove specified columns
    columns_to_remove = ["IYEAR", "DISPCODE"]
    columns_to_remove_indices = [x_train_headers.index(col) for col in columns_to_remove if col in x_train_headers]
    x_train = remove_columns_by_index(x_train, columns_to_remove_indices)
    '''

    # Step 3: Remove features with too many missing values
    valid_columns, x_train = remove_missing_values(x_train, missing_val_threshold)
    x_train_headers = [x_train_headers[i] for i in valid_columns]  # Update headers after column removal
    x_test = x_test[:, valid_columns]  # Apply the same columns to x_test
    
    # Step 4: Identify feature types
    feature_indices, _ = identify_feature_types(x_train)
    binary_columns_indices = feature_indices["binary_indices"]
    continuous_columns_indices = feature_indices["continuous_indices"]
    categorical_columns_indices = feature_indices["categorical_indices"]
    
    # Step 5: Replace binary values from 1,2 to 0,1
    x_train = replace_binary_values(x_train, binary_columns_indices)
    x_test = replace_binary_values(x_test, binary_columns_indices)
    
    # Step 7: Standardize continuous features in both train and test sets
    x_train = standardize_continuous_features(x_train, continuous_columns_indices)
    x_test = standardize_continuous_features(x_test, continuous_columns_indices)
    
    # Step 8: One-hot encode categorical features in both train and test sets
    x_train_encoded, binary_train_indices = one_hot_encode(x_train, categorical_columns_indices)
    x_test_encoded, binary_test_indices = one_hot_encode(x_test, categorical_columns_indices)

    feature_indices, _ = identify_feature_types(x_train_encoded)
    binary_columns_indices = feature_indices["binary_indices"]
    continuous_columns_indices = feature_indices["continuous_indices"]
    categorical_columns_indices = feature_indices["categorical_indices"]
    
    # Step 9: Mean imputation for missing values in both train and test sets
    x_train_imputed = mean_imputation(x_train_encoded, binary_train_indices)
    x_test_imputed = mean_imputation(x_test_encoded, binary_test_indices)
    
    # Step 10: Variance thresholding to remove low-variance features
    valid_columns, x_train_variance_filtered = variance_thresholding(x_train_imputed, variance_threshold)
    x_test_variance_filtered = x_test_imputed[:, valid_columns]
    
    # Step 11: Correlation analysis to retain relevant features
    correlation_valid_columns, x_train_correlation_filtered = correlation_analysis(
        x_train_variance_filtered, y_train, low_corr_threshold, high_corr_threshold
    )
    
    # Step 12: Statistical test analysis for feature selection
    statistical_valid_columns, x_train_statistical_filtered = statistical_test_analysis(
        x_train_variance_filtered, y_train, continuous_columns_indices, p_value_threshold
    )
    
    # Step 13: Combine correlation and statistical test results
    combined_valid_columns = sorted(set(correlation_valid_columns) | set(statistical_valid_columns))
    x_train_final = x_train_variance_filtered[:, combined_valid_columns]

    # Keep the same features in x_test than the ones selected in x_train for consistency
    x_test_final = x_test_variance_filtered[:, combined_valid_columns]

    # Step 6: Split and balance the data
    x_train_balanced, x_train_test, y_train_balanced, y_test = split_and_balance_data(
        x_train_final, y_train, test_size=test_size
    )
    
    return x_train_balanced, x_train_test, y_train_balanced, y_test, x_test_final