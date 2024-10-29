import numpy as np
from helpers import *

removal_log = []
removed_missing_values = []
removed_variance = []
removed_correlation = []

def log_and_print_removals(headers, removed_indices, step_name, reason):
    """
    Logs and prints the features removed at each step.

    Args:
        headers (list): The list of feature names.
        removed_indices (list): Indices of features that were removed at the current step.
        step_name (str): The name of the step where the features were removed.
        reason (str): The reason why the features were removed.
    """
    removed_features = [headers[i] for i in removed_indices]
    for feature in removed_features:
        removal_log.append(f"{feature} removed at {step_name}: {reason}")

    print(f"\nStep {step_name} - Reason: {reason}:")
    print(f"Removed {len(removed_features)} features: {removed_features}")

    # Function to print kept/removed features after each step
def print_feature_info(headers, kept_indices, step_name):
    """
    Prints the number of features that were kept and removed after each step, 
    along with the respective feature names.

    Args:
        headers (list): The list of feature names.
        kept_indices (list): Indices of features that were kept after the step.
        step_name (str): The name of the step for which this information is printed.
    """
    kept_features = [headers[i] for i in kept_indices]
    removed_features = [headers[i] for i in range(len(headers)) if i not in kept_indices]
    
    print(f"\nStep {step_name}:")
    print(f"Kept {len(kept_features)} features: {kept_features}")
    print(f"Removed {len(removed_features)} features: {removed_features}")

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


def identify_feature_types(data, headers, categorical_threshold=10):
    """
    Identifies the types of features in a dataset, classifying them as binary, continuous, or categorical.

    Args:
        data (np.array): The dataset as a numpy array with features as columns.
        headers (list): List of feature names corresponding to data columns.
        categorical_threshold (int): The maximum unique value count to classify a feature as categorical.

    Returns:
        dict: A dictionary with lists of binary, continuous, and categorical feature names.
        dict: A dictionary with counts of each feature type.
    """
    # Initialize counters and lists for each feature type
    binary_features = []
    continuous_features = []
    categorical_features = []

    # Analyze each feature column-wise
    for col_idx in range(data.shape[1]):
        column_data = data[:, col_idx]
        
        # Remove missing or NaN values for accurate counting
        column_data = column_data[~np.isnan(column_data)]
        
        # Get unique values
        unique_values = np.unique(column_data)
        unique_count = len(unique_values)

        # Classify features based on unique values
        if unique_count == 2 and set(unique_values) == {0, 1}:
            binary_features.append(headers[col_idx])
        elif unique_count > categorical_threshold:
            continuous_features.append(headers[col_idx])
        else:
            categorical_features.append(headers[col_idx])

    # Prepare the output dictionaries
    feature_lists = {
        "binary_features": binary_features,
        "continuous_features": continuous_features,
        "categorical_features": categorical_features,
    }

    feature_counts = {
        "binary_count": len(binary_features),
        "continuous_count": len(continuous_features),
        "categorical_count": len(categorical_features),
    }

    return feature_lists, feature_counts


def remove_missing_values(headers, data, threshold=0.25):
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
    removed_columns = np.where(missing_values > threshold)[0]

    # Store the removed features
    removed_features_names = [headers[i] for i in removed_columns]
    removed_missing_values.extend(removed_features_names)
    
    # Log removed features
    log_and_print_removals(headers, removed_columns, "1: Remove Missing Values", "Too many missing values")
    
    # Filter out the columns with too many missing values
    filtered_data = data[:, valid_columns]
    return valid_columns, filtered_data

def one_hot_encode(data, headers, categorical_columns):
    """
    Transforms categorical features into binary (one-hot encoded) features.

    Args:
        data (np.array): The dataset as a numpy array.
        headers (list): List of feature names corresponding to data columns.
        categorical_columns (list): List of indices of categorical features in the dataset.

    Returns:
        np.array: Updated dataset with one-hot encoded features.
        list: Updated headers with new binary feature names.
    """
    updated_data = []
    updated_headers = []

    for col_idx in range(data.shape[1]):
        if col_idx in categorical_columns:
            # Get unique categories for this feature
            unique_categories = np.unique(data[~np.isnan(data[:, col_idx]), col_idx])
            
            # Create binary columns for each category
            for category in unique_categories:
                binary_column = (data[:, col_idx] == category).astype(int)
                updated_data.append(binary_column)
                
                # Add new binary feature name to headers
                updated_headers.append(f"{headers[col_idx]}_{category}_encoded")
        else:
            # For non-categorical columns, add them directly
            updated_data.append(data[:, col_idx])
            updated_headers.append(headers[col_idx])

    # Stack the updated data columns horizontally
    updated_data = np.column_stack(updated_data)
    return updated_data, updated_headers


def verify_one_hot_encoding(data, binary_columns_indices):
    """
    Verifies that all binary columns contain only 0s and 1s, and that all columns
    in the dataset are either binary or continuous.

    Args:
        data (np.array): The dataset after one-hot encoding.
        binary_columns_indices (list): List of indices of columns that should be binary.

    Returns:
        bool: True if all checks pass, False otherwise.
    """
    # Check binary columns
    binary_check_passed = True
    for col in binary_columns_indices:
        unique_values = np.unique(data[:, col])
        if not np.array_equal(unique_values, [0, 1]) and not np.array_equal(unique_values, [1, 0]):
            print(f"Check failed: Column {col} has values other than 0 and 1.")
            binary_check_passed = False

    # Check non-binary columns are continuous
    continuous_check_passed = True
    for col in range(data.shape[1]):
        if col not in binary_columns_indices:
            unique_values = np.unique(data[:, col])
            if len(unique_values) < 10:  # Continuous data should have a broad range of unique values
                print(f"Check failed: Column {col} appears to have low unique values, suggesting it may not be continuous.")
                continuous_check_passed = False

    # Summary
    if binary_check_passed and continuous_check_passed:
        print("All checks passed: The dataset has only binary or continuous features.")
        return True
    else:
        print("Some checks failed. Please review the warnings above.")
        return False
    

def mean_imputation(data, binary_columns):
    """
    Imputes missing values in the dataset by replacing them with the mean of each column.
    For binary columns, the mean is rounded to 0 or 1.

    Args:
        data (np.array): The dataset with missing values (numpy array with NaN for missing values).
        binary_columns (list): List of indices representing binary columns.

    Returns:
        np.array: The dataset with missing values replaced by column means or rounded means for binary features.
    """
    col_means = np.nanmean(data, axis=0)
    inds = np.where(np.isnan(data))
    
    # For binary columns, round the mean to 0 or 1
    for col in binary_columns:
        col_means[col] = round(col_means[col])

    # Replace NaNs with the appropriate mean (rounded for binary columns)
    data[inds] = np.take(col_means, inds[1])
    
    return data


def variance_thresholding(headers, data, threshold=0.01):
    """
    Removes features that have variance below a specified threshold.

    Args:
        headers (list): The list of feature names.
        data (np.array): The dataset (numpy array) with features as columns.
        threshold (float): The minimum variance threshold (default is 1%).

    Returns:
        tuple: A tuple containing:
            - valid_columns (list): The indices of the features that were kept.
            - filtered_data (np.array): The filtered dataset with only the valid columns.
    """
    # Compute variance for each column
    variances = np.nanvar(data, axis=0)
    valid_columns = np.where(variances >= threshold)[0]
    removed_columns = np.where(variances < threshold)[0]

    # Store the removed features
    removed_features_names = [headers[i] for i in removed_columns]
    removed_variance.extend(removed_features_names)
    
    # Log removed features
    log_and_print_removals(headers, removed_columns, "2: Variance Thresholding", "Low variance")
    
    # Keep only columns with variance above the threshold
    filtered_data = data[:, valid_columns]
    return valid_columns, filtered_data


def correlation_analysis(headers, x_data, y_data, low_threshold=0.05, high_threshold=0.9):
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
    removed_columns = []

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
            else:
                removed_columns.append(i)
        else:
            removed_columns.append(i)

    # Log removed features
    log_and_print_removals(headers, removed_columns, "3: Correlation Analysis", "Low or high correlation")

    # Store the removed features
    removed_features_names = [headers[i] for i in removed_columns]
    removed_correlation.extend(removed_features_names)

    # Filter valid columns
    filtered_data = x_data[:, valid_columns]
    return valid_columns, filtered_data


def statistical_test_analysis(headers, x_data, y_data, p_value_threshold=0.05):
    """
    Retains features that have a statistically significant association with the outcome variable (y_data),
    using an F-test for continuous features and a Chi-Square test for categorical features.

    Args:
        headers (list): The list of feature names.
        x_data (np.array): The feature dataset (numpy array) with features as columns.
        y_data (np.array): The outcome variable (labels) to test associations with.
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

        # Check if the feature is continuous or categorical
        unique_vals = np.unique(feature_clean)
        if len(unique_vals) > 10:  # Assume continuous if more than 10 unique values
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
            contingency_table = np.array([[(feature_clean == val).sum() & (y_data_clean == label).sum() for label in unique_y] for val in unique_vals])
            chi_square_statistic = np.sum((contingency_table - np.mean(contingency_table))**2 / np.mean(contingency_table))
            p_value = 1 - (1 / (1 + chi_square_statistic))  # Simplified approximation

        # Determine if the feature is statistically significant
        if p_value < p_value_threshold:
            valid_columns.append(i)
        else:
            removed_columns.append(i)

    # Log removed features
    log_and_print_removals(headers, removed_columns, "4: Statistical Test Analysis", "Not statistically significant")

    # Filter valid columns
    filtered_data = x_data[:, valid_columns]
    return valid_columns, filtered_data



def Preprocess_data(file_path, x_train, x_test, y_train, low_corr_threshold=0.05, high_corr_threshold=0.9, variance_threshold=0.01, p_value_threshold=0.05, missing_val_threshold=0.25):
    """
    Preprocesses the dataset by sequentially applying all preprocessing steps: missing values removal, 
    one-hot encoding, mean imputation, variance thresholding, correlation analysis, and statistical test analysis.

    Args:
        file_path (str): Path to the dataset folder.
        categorical_columns (list): List of indices of categorical features for one-hot encoding.
        binary_features (list): List of indices of known binary features.
        low_corr_threshold (float): Minimum Pearson correlation threshold for correlation analysis.
        high_corr_threshold (float): Maximum Pearson correlation threshold for correlation analysis.
        variance_threshold (float): Minimum variance threshold to retain features.
        p_value_threshold (float): Significance level threshold for statistical analysis.
        missing_val_threshold (float): Maximum allowed percentage of missing values.

    Returns:
        np.array: Preprocessed x_train, x_test, and y_train datasets.
    """
    x_train_headers = extract_headers(file_path + "x_train.csv")

    # Step 1: Remove features with too many missing values
    valid_columns, x_train = remove_missing_values(x_train_headers, x_train, missing_val_threshold)
    x_train_headers = [x_train_headers[i] for i in valid_columns]
    x_test = x_test[:, valid_columns]

    # Step 2: Identify Feature Types
    feature_lists, feature_counts = identify_feature_types(x_train, x_train_headers)
    binary_features = feature_lists["binary_features"]
    categorical_features = feature_lists["categorical_features"]

    # Step 3: One-Hot Encoding for categorical features
    categorical_columns_filtered = [i for i, header in enumerate(x_train_headers) if header in categorical_features]
    x_train_encoded, x_train_encoded_headers = one_hot_encode(x_train, x_train_headers, categorical_columns_filtered)
    x_test_encoded, _ = one_hot_encode(x_test, x_train_headers, categorical_columns_filtered)

    # Step 4: Verify One-Hot Encoding
    binary_columns = [i for i, header in enumerate(x_train_encoded_headers) if "_encoded" in header or header in binary_features]
    verify_one_hot_encoding(x_train_encoded, binary_columns)

    # Step 5: Mean Imputation for Missing Values
    x_train_imputed = mean_imputation(x_train_encoded, binary_columns)
    x_test_imputed = mean_imputation(x_test_encoded, binary_columns)

    # Step 6: Variance Thresholding
    valid_columns, x_train_variance_filtered = variance_thresholding(x_train_encoded_headers, x_train_imputed, variance_threshold)
    x_train_variance_filtered_headers = [x_train_encoded_headers[i] for i in valid_columns]
    x_test_variance_filtered = x_test_imputed[:, valid_columns]

    # Step 7: Correlation Analysis
    correlation_valid_columns, x_train_correlation_filtered = correlation_analysis(
        x_train_variance_filtered_headers, x_train_variance_filtered, y_train, low_corr_threshold, high_corr_threshold
    )
    x_train_correlation_filtered_headers = [x_train_variance_filtered_headers[i] for i in correlation_valid_columns]
    x_test_correlation_filtered = x_test_variance_filtered[:, correlation_valid_columns]

    # Step 8: Statistical Test Analysis
    statistical_valid_columns, x_train_statistical_filtered = statistical_test_analysis(
        x_train_variance_filtered_headers, x_train_variance_filtered, y_train, p_value_threshold
    )
    x_train_statistical_filtered_headers = [x_train_variance_filtered_headers[i] for i in statistical_valid_columns]
    x_test_statistical_filtered = x_test_variance_filtered[:, statistical_valid_columns]

    # Combine results from both correlation and statistical tests (union of both selections)
    combined_valid_columns = sorted(set(correlation_valid_columns) | set(statistical_valid_columns))
    x_train_final = x_train_variance_filtered[:, combined_valid_columns]
    x_train_final_headers = [x_train_variance_filtered_headers[i] for i in combined_valid_columns]
    x_test_final = x_test_variance_filtered[:, combined_valid_columns]
    
    return x_train_final_headers, x_train_final, x_test_final, y_train