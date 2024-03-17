

# A/B Testing Script

# Importing necessary libraries

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import anderson

# Load the data from the CSV file
data = pd.read_csv("user-data.csv")

# Display the first few rows of the dataset
print(data.head())

# Separate the data by variant A and variant B
variant_a = data[data["Variant"] == "A"]
variant_b = data[data["Variant"] == "B"]

# Plot the distributions of the metrics by variant using histograms
metrics = ["Clicks on media", "Time on Page (sec)"]
for metric in metrics:
    plt.figure(figsize=(8, 6))
    sns.histplot(x=metric, hue="Variant", data=data, multiple="dodge", stat="count")
    plt.title(f"Distribution of {metric} by Variant")
    plt.xlabel(metric)
    plt.ylabel("Count")
    plt.legend("AB")
    plt.show()

# Function to interpret normality of distributions
def interpret_normality(metric, variant_name, mean, skewness, kurtosis):
    """
    Interpret the normality of the distribution for a given metric and variant.

    Args:
        metric (str): The name of the metric.
        variant_name (str): The name of the variant.
        mean (float): The mean of the distribution.
        skewness (float): The skewness of the distribution.
        kurtosis (float): The kurtosis of the distribution.
    """
    # Print interpretation header
    print("\nInterpretation:")
    
    # Check skewness of the distribution
    if abs(skewness) < 0.5:
        print(f"\t{variant_name} for {metric} is approximately symmetric.")
    else:
        print(f"\t{variant_name} for {metric} is skewed.")

    # Check kurtosis of the distribution
    if abs(kurtosis - 3) < 0.5:
        print(f"\t{variant_name} for {metric} has approximately normal kurtosis.")
    else:
        print(f"\t{variant_name} for {metric} deviates from normal kurtosis.")

# Loop through metrics and variants to calculate statistics and interpret normality
for metric in metrics:
    print(f"Metric: {metric}")

    # Calculate statistics for variant A
    mean_a = variant_a[metric].mean()
    skewness_a = stats.skew(variant_a[metric])
    kurtosis_a = stats.kurtosis(variant_a[metric])

    # Print results for variant A
    print(f"\tVariant A:")
    print(f"\t\tMean: {mean_a:.4f}")
    print(f"\t\tSkewness: {skewness_a:.4f}")
    print(f"\t\tKurtosis: {kurtosis_a:.4f}")

    # Calculate statistics for variant B
    mean_b = variant_b[metric].mean()
    skewness_b = stats.skew(variant_b[metric])
    kurtosis_b = stats.kurtosis(variant_b[metric])

    # Print results for variant B
    print(f"\tVariant B:")
    print(f"\t\tMean: {mean_b:.4f}")
    print(f"\t\tSkewness: {skewness_b:.4f}")
    print(f"\t\tKurtosis: {kurtosis_b:.4f}")

    # Interpret normality for variant A
    interpret_normality(metric, "Variant A", mean_a, skewness_a, kurtosis_a)

    # Interpret normality for variant B
    interpret_normality(metric, "Variant B", mean_b, skewness_b, kurtosis_b)

    print("-" * 30)  # Optional separator between metrics

# Function to interpret Shapiro-Wilk test results
def evaluate_shapiro_wilk(pval, alpha):
    """
    Interpret the results of the Shapiro-Wilk test based on the significance level.

    Args:
        pval (float): The p-value from the Shapiro-Wilk test.
        alpha (float): The significance level for hypothesis testing.

    Returns:
        str: Interpretation of the Shapiro-Wilk test result.
    """
    if pval > alpha:
        return "Data likely follows a normal distribution."
    else:
        return "Data may not be normally distributed."

# Function to evaluate Shapiro-Wilk test results for each metric and variant
def evaluate_shapiro_wilk_results(metrics, variant_a, variant_b, alpha=0.05):
    """
    Evaluate the results of the Shapiro-Wilk test for each metric and variant.

    Args:
        metrics (list): List of metric names.
        variant_a (dict): Dictionary containing variant A data for each metric.
        variant_b (dict): Dictionary containing variant B data for each metric.
        alpha (float, optional): The significance level for hypothesis testing. Defaults to 0.05.

    Returns:
        dict: Dictionary containing the evaluation results for each metric and variant.
    """
    evaluation_results = {}

    for metric in metrics:
        # Test statistic and p-value for variant A
        stat_a, pval_a = stats.shapiro(variant_a[metric])

        # Test statistic and p-value for variant B
        stat_b, pval_b = stats.shapiro(variant_b[metric])

        # Evaluate Shapiro-Wilk test results using separate function
        evaluation_a = evaluate_shapiro_wilk(pval_a, alpha)
        evaluation_b = evaluate_shapiro_wilk(pval_b, alpha)

        # Store evaluation results
        evaluation_results[metric] = {
            'Variant A': {'Statistic': stat_a, 'P-value': pval_a, 'Interpretation': evaluation_a},
            'Variant B': {'Statistic': stat_b, 'P-value': pval_b, 'Interpretation': evaluation_b}
        }

    return evaluation_results

# Call the function to evaluate the Shapiro-Wilk test results
evaluation_results = evaluate_shapiro_wilk_results(metrics, variant_a, variant_b)

# Print evaluation results
for metric, evaluations in evaluation_results.items():
    print(f"Metric: {metric}")
    for variant, result in evaluations.items():
        print(f"\t{variant}:")
        print(f"\t\tShapiro-Wilk Statistic: {result['Statistic']:.4f}")
        print(f"\t\tP-value: {result['P-value']:.4f}")
        print(f"\t\tInterpretation: {result['Interpretation']}")
    print("-" * 30) # Optional separator between metrics

# Function to perform the Anderson-Darling test for normality
def perform_anderson_darling_test(data):
    """
    Perform the Anderson-Darling test for normality on the given data.

    Args:
        data (array-like): The data to be tested for normality.

    Returns:
        tuple: A tuple containing the Anderson-Darling test statistic, critical values, and significance levels.
    """
    return anderson(data, dist='norm')

# Function to evaluate the normality of data based on the Anderson-Darling test
def evaluate_normality(ad_statistic, ad_critical_values):
    """
    Evaluate the normality of the data based on the Anderson-Darling test results.

    Args:
        ad_statistic (float): The Anderson-Darling test statistic.
        ad_critical_values (array-like): The critical values for the Anderson-Darling test.

    Returns:
        bool: True if the data is normally distributed based on the Anderson-Darling test, False otherwise.
    """
    return ad_statistic < ad_critical_values[2]  # Using the 5% significance level

# Iterate over each metric to perform the Anderson-Darling test
for metric in metrics:
    # Perform the Anderson-Darling test for variant A
    ad_statistic_a, ad_critical_values_a, ad_significance_levels_a = perform_anderson_darling_test(variant_a[metric])
    
    # Perform the Anderson-Darling test for variant B
    ad_statistic_b, ad_critical_values_b, ad_significance_levels_b = perform_anderson_darling_test(variant_b[metric])

    # Print the test results for variant A
    print(f"Metric: {metric}")
    print("Variant A:")
    print(f"\tAnderson-Darling Test Statistic: {ad_statistic_a:.4f}")
    print(f"\tCritical Values: {ad_critical_values_a}")
    print(f"\tSignificance Levels: {ad_significance_levels_a}")
    print(f"\tIs normally distributed: {evaluate_normality(ad_statistic_a, ad_critical_values_a)}")

    # Print the test results for variant B
    print("Variant B:")
    print(f"\tAnderson-Darling Test Statistic: {ad_statistic_b:.4f}")
    print(f"\tCritical Values: {ad_critical_values_b}")
    print(f"\tSignificance Levels: {ad_significance_levels_b}")
    print(f"\tIs normally distributed: {evaluate_normality(ad_statistic_b, ad_critical_values_b)}")
    print("-" * 50)  # Separator between metrics

# Function to perform a two-sample t-test for the given metrics
def perform_t_test(metric_data_a, metric_data_b):
    """
    Perform a two-sample t-test for the given metrics.

    Args:
        metric_data_a (array-like): Data for variant A for the given metric.
        metric_data_b (array-like): Data for variant B for the given metric.

    Returns:
        tuple: A tuple containing the t-statistic, p-value, mean, standard deviation, and sample size
                for the two-sample t-test.
    """
    import numpy as np  # Import numpy within the function

    # Calculate mean, standard deviation, and sample size
    mean_a = np.mean(metric_data_a)
    mean_b = np.mean(metric_data_b)
    std_a = np.std(metric_data_a)
    std_b = np.std(metric_data_b)
    size_a = len(metric_data_a)
    size_b = len(metric_data_b)

    # Perform t-test
    t_statistic, p_value = ttest_ind(metric_data_a, metric_data_b)

    return t_statistic, p_value, mean_a, mean_b, std_a, std_b, size_a, size_b

# Iterate over each metric to perform the two-sample t-test
for metric in metrics:
    # Extract data for variant A and variant B for the current metric
    metric_data_a = variant_a[metric]
    metric_data_b = variant_b[metric]

    # Perform the two-sample t-test
    t_statistic, p_value, mean_a, mean_b, std_a, std_b, size_a, size_b = perform_t_test(metric_data_a, metric_data_b)

    # Print the test results
    print(f"Metric: {metric}")
    print(f"\tVariant A: Mean={mean_a:.4f}, Std Dev={std_a:.4f}, Sample Size={size_a}")
    print(f"\tVariant B: Mean={mean_b:.4f}, Std Dev={std_b:.4f}, Sample Size={size_b}")
    print(f"\tTwo-sample t-test - t-statistic: {t_statistic:.4f}, p-value: {p_value:.4f}")
    print(f"\tConclusion: {'Reject null hypothesis' if p_value < 0.05 else 'Fail to reject null hypothesis'}")
    print("-" * 50)  # Separator between metrics
