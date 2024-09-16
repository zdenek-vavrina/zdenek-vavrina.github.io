# Quick Sum Up of Statistical Tests

---

### Introduction
This summary provides a quick reference for conducting various statistical tests in R and Python. Each code snippet demonstrates how to perform the test and interpret the results. Ensure that your data meets the assumptions of the chosen test for accurate analysis.

## Statistical Tests

### Descriptive Statistics
- *Purpose: Provides an overview of data characteristics and summarizes data sets.*
-  **[Descriptive Statistics](#descriptive-statistics-1)**

### Comparing Means
- *Purpose: Tests comparing the means of one or more groups to determine significant differences.*

### Statistical Tests for Comparing Group Means

- #### Between Two Groups
  - **[One Sample T-test](#one-sample-t-test)**
  - **[Two Sample T-test](#two-sample-t-test)**
  - **[Paired Sample T-test](#paired-sample-t-test)**
  - **[Paired Z-test](#paired-z-test)**

- #### Between More Than Two Groups
  - **[One-Way ANOVA](#one-way-anova)**
  - **[Two-Way ANOVA](#two-way-anova)**

- #### Non-Parametric Tests for Comparing Means
  - **[Kruskal-Wallis H-test](#kruskal-wallis-h-test)**
  - **[Friedman Test](#friedman-test)**

### Comparing Distributions
- *Purpose: Non-parametric tests comparing distributions of two or more groups.*

- **[Wilcoxon Signed Rank Test](#wilcoxon-signed-rank-test)**
- **[Mann-Whitney U Test](#mann-whitney-u-test)**

### Categorical Data Analysis
- *Purpose: Tests for independence and associations between categorical variables.*

- **[Chi-Square Test for Independence](#chi-square-test-for-independence)**
- **[Fisher's Exact Test](#fishers-exact-test-detailed-use-case)**

### *Post hoc* Multiple Comparisons
- *Purpose: Adjusts for multiple comparisons and tests to control Type I error rates.*

- **[Tukey's Honestly Significant Difference (HSD) Test](#tukeys-honestly-significant-difference-hsd-test)**
- **[Bonferroni Correction](#bonferroni-correction)**

### Assumption Checks
- *Purpose*: Tests to check assumptions such as normality and equal variances required for parametric tests.

- **[Levene's Test](#levenes-test)**
- **[Testing of Normality Distribution for Paired Test](#testing-of-normality-distribution-for-paired-test)**

### Choosing the Right Statistical Test
- *Purpose*: Overview of decision tree analysis methods and their applications.

- **[Decision tree](#decision-tree)**

---

## [Descriptive Statistics](#descriptive-statistics-1)
- Descriptive statistics summarize and describe the main features of a dataset
- Includes such measures as mean, median, standard deviation, and range.

### R Code (for data.frame)
```r
# Load necessary library
library(dplyr)

# Example data: replace with your actual data
data <- data.frame(
  group = factor(rep(c("Group1", "Group2", "Group3"), each = 10)),
  value = rnorm(30)  # Generates 30 random normal values
)

# Descriptive statistics
summary_stats <- data %>%
  group_by(group) %>%
  summarise(
    mean = mean(value),
    median = median(value),
    sd = sd(value),
    min = min(value),
    max = max(value),
    n = n()
  )

# Print the results
print(summary_stats)
```

### R Code (for tibble)
```r
# Load necessary libraries
library(dplyr)
library(tibble)

# Example data: replace with your actual data
data <- tibble(
  group = factor(rep(c("Group1", "Group2", "Group3"), each = 10)),
  value = rnorm(30)  # Generates 30 random normal values
)

# Descriptive statistics
summary_stats <- data %>%
  group_by(group) %>%
  summarise(
    mean = mean(value),
    median = median(value),
    sd = sd(value),
    min = min(value),
    max = max(value),
    n = n()
  )

# Print the results
print(summary_stats)
```

### Python Code (for pandas DataFrame)
```python
import pandas as pd
import numpy as np

# Example data: replace with your actual data
data = pd.DataFrame({
    'group': np.repeat(['Group1', 'Group2', 'Group3'], 10),
    'value': np.random.randn(30)  # Generates 30 random normal values
})

# Descriptive statistics
summary_stats = data.groupby('group').agg(
    mean=('value', 'mean'),
    median=('value', 'median'),
    sd=('value', 'std'),
    min=('value', 'min'),
    max=('value', 'max'),
    n=('value', 'size')
).reset_index()

# Print the results
print(summary_stats)
```
[↑ Back to Top](#statistical-tests)

## [One Sample T-test](#one-sample-t-test)
- **Description**: Compares a sample mean to a known population mean when the population standard deviation is unknown or the sample size is small.
- **Assumptions**:
  - Data is independent.
  - Parent population doesn't need normal distribution.
  - Sample means need to be normally distributed.
  - Ideally, sample size < 30.
- **Null Hypothesis**: The sample mean is equal to the population mean.

### R Code
```r
# Sample data: replace '...' with your actual data
sample_data <- c(...) 

# Known population mean
population_mean <- 50 

# Perform the one-sample t-test
t_test_result <- t.test(sample_data, mu = population_mean)

# Display the results
print(t_test_result)
```

### Python Code
```python
from scipy import stats

# Sample data: replace '...' with your actual data
sample_data = [...] 

# Known population mean
population_mean = 50

# Perform the one-sample t-test
t_stat, p_value = stats.ttest_1samp(sample_data, population_mean)

# Print the results
print(f"T-statistic: {t_stat}, P-value: {p_value}")
```

## [Z-test](#z-test)
- **Description**: Compares a sample mean to a known population mean when the population standard deviation is known and the sample size is large.
- **Assumptions**:
  - Data is independent.
  - Parent population is normally distributed or sample size is large.
  - Population standard deviation is known.
  - Ideally, sample size > 30.
- **Null Hypothesis**: The sample mean is equal to the population mean.

### R Code
```r
# Sample data
sample_mean <- 55
population_mean <- 50
population_sd <- 10
sample_size <- 100

# Perform the Z-test
z_test_result <- z.test(x = sample_mean, mu = population_mean, sigma.x = population_sd, n.x = sample_size)

# Display the results
print(z_test_result)
```

### Python Code
```python
from statsmodels.stats import ztest

# Sample data: replace '...' with your actual data
sample_data = [...] 

# Known population mean and standard deviation
population_mean = 50
population_sd = 10

# Perform the Z-test
z_stat, p_value = ztest(sample_data, value=population_mean, ddof=0)

# Print the results
print(f"Z-statistic: {z_stat}, P-value: {p_value}")
```
[↑ Back to Top](#statistical-tests)

## [Two Sample T-test](#two-sample-t-test)
- **Description**: Compares the means of two independent samples when population standard deviations are unknown.
- **Assumptions**:
  - Data is independent.
  - Parent populations don't need normal distribution.
  - Sample means need to be normally distributed.
  - Assumes equal variances (use Welch's test if not) - check by Levene's test.
  - Ideally, sample size < 30 per group.
- **Null Hypothesis**: The means of the two independent samples are equal.

### R Code
```r
# Sample data for two groups
group1 <- c(...) # Replace with your data for group 1
group2 <- c(...) # Replace with your data for group 2

# Perform the two-sample t-test
t_test_result <- t.test(group1, group2)

# Display the results
print(t_test_result)
```

### Python Code
```python
from scipy import stats

# Sample data for two groups: replace '...' with your actual data
group1 = [...] 
group2 = [...] 

# Perform the two-sample t-test
t_stat, p_value = stats.ttest_ind(group1, group2)

# Print the results
print(f"T-statistic: {t_stat}, P-value: {p_value}")
```
[↑ Back to Top](#statistical-tests)

## [Paired Sample T-test](#paired-sample-t-test)
- **Description**: Compares the means of two related samples (e.g., before and after) when population standard deviation is unknown.
- **Assumptions**:
  - Data is dependent (paired observations).
  - Differences between pairs should be normally distributed.
  - No assumption of equal variances.
  - Ideally, sample size < 30 pairs.
- **Null Hypothesis**: The mean difference between paired samples is zero.

### R Code
```r
# Paired data
before <- c(...) # Replace with your data for before
after <- c(...) # Replace with your data for after

# Perform the paired t-test
paired_t_test_result <- t.test(before, after, paired = TRUE)

# Display the results
print(paired_t_test_result)
```

### Python Code
```python
from scipy import stats

# Paired sample data: replace '...' with your actual data
before = [...] 
after = [...] 

# Perform the paired t-test
t_stat, p_value = stats.ttest_rel(before, after)

# Print the results
print(f"T-statistic: {t_stat}, P-value: {p_value}")
```
[↑ Back to Top](#statistical-tests)

## [Paired Z-test](#paired-z-test)
- **Description**: Compares the means of two related samples (e.g., before and after) when the population standard deviation is known, and the sample size is large (typically n > 30).
- **Assumptions**:
  - Data is dependent (paired observations).
  - Differences between pairs should be normally distributed.
  - The population standard deviation is known.
- **Null Hypothesis**: The means of the two related samples are equal.

### R Code
```r
# Calculate the mean difference
mean_diff <- mean(after - before)

# Calculate the standard error of the mean difference
sd_diff <- 10 # known population standard deviation
n <- length(before)
std_error <- sd_diff / sqrt(n)

# Compute the Z statistic
z_stat <- mean_diff / std_error

# Compute the p-value
p_value <- 2 * pnorm(-abs(z_stat))

# Output results
list(Z_Statistic = z_stat, P_Value = p_value)
```

### Python Code
```python
import numpy as np
from scipy import stats

# Sample data
before = [...] # Replace with your data
after = [...] # Replace with your data
sd_diff = 10  # Known population standard deviation

# Calculate mean difference
mean_diff = np.mean(np.array(after) - np.array(before))

# Calculate the standard error
n = len(before)
std_error = sd_diff / np.sqrt(n)

# Compute the Z statistic
z_stat = mean_diff / std_error

# Compute the p-value
p_value = 2 * stats.norm.cdf(-abs(z_stat))

# Print the results
print(f"Z-Statistic: {z_stat}, P-value: {p_value}")
```

## [One-Way ANOVA](#one-way-anova)
- **Description**: Tests for differences in means among three or more independent groups.
- **Assumptions**:
  - Data is independent.
  - Data is normally distributed within groups.
  - Homogeneity of variances across groups (check with Levene's test).
- **Null Hypothesis**: All group means are equal.

### R Code
```r
# Data for three groups
group1 <- c(...) # Replace with your data for group 1
group2 <- c(...) # Replace with your data for group 2
group3 <- c(...) # Replace with your data for group 3

# Combine data into a data frame
data <- data.frame(
  value = c(group1, group2, group3),
  group = factor(rep(c("Group1", "Group2", "Group3"), times = c(length(group1), length(group2), length(group3))))
)

# Perform One-Way ANOVA
anova_result <- aov(value ~ group, data = data)

# Display the results
summary(anova_result)
```

### Python Code
```python
from scipy import stats
import pandas as pd

# Data for three groups: replace '...' with your actual data
group1 = [...] 
group2 = [...] 
group3 = [...] 

# Combine data into a DataFrame
data = pd.DataFrame({
    'value': group1 + group2 + group3,
    'group': ['Group1']*len(group1) + ['Group2']*len(group2) + ['Group3']*len(group3)
})

# Perform One-Way ANOVA
anova_result = stats.f_oneway(group1, group2, group3)

# Print the results
print(f"F-statistic: {anova_result.statistic}, P-value: {anova_result.pvalue}")
```
[↑ Back to Top](#statistical-tests)

## [Kruskal-Wallis H-test](#kruskal-wallis-h-test)
- **Description**: Non-parametric test for comparing medians among three or more independent groups.
- **Assumptions**:
  - Data is independent.
  - Data does not need to be normally distributed.
  - Sample sizes can be unequal.
- **Null Hypothesis**: All group distributions are equal.

### R Code
```r
# Data for three groups
group1 <- c(...) # Replace with your data for group 1
group2 <- c(...) # Replace with your data for group 2
group3 <- c(...) # Replace with your data for group 3

# Combine data into a data frame
data <- data.frame(
  value = c(group1, group2, group3),
  group = factor(rep(c("Group1", "Group2", "Group3"), times = c(length(group1), length(group2), length(group3))))
)

# Perform Kruskal-Wallis H-test
kruskal_result <- kruskal.test(value ~ group, data = data)

# Display the results
print(kruskal_result)
```

### Python Code
```python
from scipy import stats
import pandas as pd

# Data for three groups: replace '...' with your actual data
group1 = [...] 
group2 = [...] 
group3 = [...] 

# Combine data into a DataFrame
data = pd.DataFrame({
    'value': group1 + group2 + group3,
    'group': ['Group1']*len(group1) + ['Group2']*len(group2) + ['Group3']*len(group3)
})

# Perform Kruskal-Wallis H-test
kruskal_result = stats.kruskal(group1, group2, group3)

# Print the results
print(f"Test statistic: {kruskal_result.statistic}, P-value: {kruskal_result.pvalue}")
```

Certainly. Here's the continuation of the statistical tests summary:

## [Two-Way ANOVA](#two-way-anova)
- **Description**: Tests for differences in means between groups when there are two independent variables. It assesses the impact of both factors on the dependent variable and also checks for interaction effects between the factors.
- **Assumptions**: 
  - The data is normally distributed.
  - Homogeneity of variances across groups.
  - The observations are independent.
- **Null Hypothesis**: There is no effect of each factor on the dependent variable, and there is no interaction effect between the factors.

### R Code
```r
# Load necessary library
library(dplyr)
library(ggplot2)

# Example data: replace with your actual data
data <- data.frame(
  factor1 = factor(rep(c("Level1", "Level2"), each = 15)),
  factor2 = factor(rep(c("A", "B", "C"), times = 10)),
  response = rnorm(30)  # Generates 30 random normal values
)

# Perform Two-Way ANOVA
anova_results <- aov(response ~ factor1 * factor2, data = data)

# Summary of the ANOVA
summary(anova_results)
```

### Python Code
```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Example data: replace with your actual data
data = pd.DataFrame({
    'factor1': np.repeat(['Level1', 'Level2'], 15),
    'factor2': np.tile(['A', 'B', 'C'], 10),
    'response': np.random.randn(30)  # Generates 30 random normal values
})

# Perform Two-Way ANOVA
model = ols('response ~ C(factor1) * C(factor2)', data=data).fit()
anova_results = sm.stats.anova_lm(model, typ=2)

# Summary of the ANOVA
print(anova_results)
```

## [Friedman Test](#friedman-test)
- **Description**: A non-parametric test used to detect differences in treatments across multiple test attempts. It is used when the same subjects are used for each treatment, making it suitable for repeated measures with more than two conditions.
- **Assumptions**:
  - The same subjects are used for all treatments (repeated measures).
  - The data is at least ordinal (ranks are sufficient).
  - The test is used for more than two related samples.
- **Null Hypothesis**: There are no differences in the treatments; any observed differences are due to random variation.

### R Code
```r
# Example of Friedman's Test in R
# Data should be in a long format with columns for subject, treatment, and response

# Load necessary library
library(dplyr)

# Generate theoretical example data
data <- data.frame(
  subject = factor(rep(1:10, each = 3)),
  treatment = factor(rep(c("A", "B", "C"), times = 10)),
  response = c(sample(1:10, 30, replace = TRUE))
)

# Perform Friedman's Test
friedman_test <- friedman.test(response ~ treatment | subject, data = data)

# Print results
print(friedman_test)
```

### Python Code
```python
# Example of Friedman's Test in Python
import pandas as pd
from scipy.stats import friedmanchisquare

# Generate theoretical example data
data = pd.DataFrame({
    'subject': list(range(1, 11)) * 3,
    'treatment': ['A'] * 10 + ['B'] * 10 + ['C'] * 10,
    'response': list(range(1, 11)) * 3
})

# Prepare data for Friedman's Test
data_pivot = data.pivot(index='subject', columns='treatment', values='response')
data_array = data_pivot.to_numpy()

# Perform Friedman's Test
stat, p_value = friedmanchisquare(*data_array.T)

# Print results
print(f"Friedman's test statistic: {stat}")
print(f"P-value: {p_value}")
```

## [Wilcoxon Signed Rank Test](#wilcoxon-signed-rank-test)
- **Description**: Non-parametric test comparing two related samples or repeated measurements to assess if their population mean ranks differ.
- **Assumptions**:
  - Data is paired and the differences between pairs should be symmetric.
  - No assumption of normality.
- **Null Hypothesis**: The median of the differences between paired samples is zero.

### R Code
```r
# Paired data
before <- c(...) # Replace with your data for before
after <- c(...) # Replace with your data for after

# Perform the Wilcoxon signed-rank test
wilcoxon_result <- wilcox.test(before, after, paired = TRUE)

# Display the results
print(wilcoxon_result)
```

### Python Code
```python
from scipy import stats

# Paired sample data: replace '...' with your actual data
before = [...] 
after = [...] 

# Perform the Wilcoxon signed-rank test
wilcoxon_result = stats.wilcoxon(before, after)

# Print the results
print(f"Test statistic: {wilcoxon_result.statistic}, P-value: {wilcoxon_result.pvalue}")
```

## [Mann-Whitney U Test](#mann-whitney-u-test)
- **Description**: Non-parametric test comparing the distributions of two independent samples.
- **Assumptions**:
  - Data is independent.
  - Data does not need to be normally distributed.
- **Null Hypothesis**: The distributions of the two samples are equal.

### R Code
```r
# Sample data for two groups
group1 <- c(...) # Replace with your data for group 1
group2 <- c(...) # Replace with your data for group 2

# Perform the Mann-Whitney U test
mann_whitney_result <- wilcox.test(group1, group2)

# Display the results
print(mann_whitney_result)
```

### Python Code
```python
from scipy import stats

# Sample data for two groups: replace '...' with your actual data
group1 = [...] 
group2 = [...] 

# Perform the Mann-Whitney U test
mann_whitney_result = stats.mannwhitneyu(group1, group2)

# Print the results
print(f"U-statistic: {mann_whitney_result.statistic}, P-value: {mann_whitney_result.pvalue}")
```

## [Chi-Square Test for Independence](#chi-square-test-for-independence)
- **Description**: Tests for independence between two categorical variables in a contingency table.
- **Assumptions**:
  - Data is categorical.
  - Observations are independent.
  - Expected frequencies should be sufficiently large (usually at least 5).
- **Null Hypothesis**: The variables are independent.

### R Code
```r
# Example data
data <- matrix(c(10, 20, 30, 40), nrow = 2)

# Perform the Chi-Square Test
chi_square_result <- chisq.test(data)

# Display the results
print(chi_square_result)
```

### Python Code
```python
from scipy import stats
import numpy as np

# Example data: replace with your actual data
data = np.array([[10, 20], [30, 40]])

# Perform the Chi-Square Test
chi2_stat, p_value, dof, expected = stats.chi2_contingency(data)

# Print the results
print(f"Chi-square statistic: {chi2_stat}, P-value: {p_value}")
```

## [Fisher's Exact Test](#fishers-exact-test-detailed-use-case)
- **Description**: Tests for independence in a 2x2 contingency table, suitable for small sample sizes.
- **Assumptions**:
  - Data is categorical.
  - Observations are independent.
  - No assumption of minimum expected cell count.
- **Null Hypothesis**: The variables are independent.

### R Code
```r
# Example data: replace with your actual data
data <- matrix(c(10, 20, 30, 40), nrow = 2)

# Perform Fisher's Exact Test
fisher_result <- fisher.test(data)

# Display the results
print(fisher_result)
```

### Python Code
```python
from scipy import stats
import numpy as np

# Example data: replace with your actual data
data = np.array([[10, 20], [30, 40]])

# Perform Fisher's Exact Test
fisher_result = stats.fisher_exact(data)

# Print the results
print(f"Odds ratio: {fisher_result[0]}, P-value: {fisher_result[1]}")
```

## [Tukey's Honestly Significant Difference (HSD) Test](#tukeys-honestly-significant-difference-hsd-test)
- **Description**: Post-hoc test following ANOVA to find which group means are significantly different.
- **Assumptions**:
  - ANOVA assumptions are met.
  - Equal variances across groups.
- **Null Hypothesis**: The means of all group pairs are equal.

### R Code
```r
# Example data
data <- data.frame(
  value = c(...),  # Replace with your data
  group = factor(c(...))  # Replace with your group labels
)

# Perform One-Way ANOVA
anova_result <- aov(value ~ group, data = data)

# Perform Tukey's HSD test
tukey_result <- TukeyHSD(anova_result)

# Display the results
print(tukey_result)
```

### Python Code
```python
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd

# Example data
data = pd.DataFrame({
    'value': [...],  # Replace with your data
    'group': [...]   # Replace with your group labels
})

# Perform One-Way ANOVA
anova_model = sm.formula.ols('value ~ group', data=data).fit()
anova_result = sm.stats.anova_lm(anova_model, typ=2)

# Perform Tukey's HSD test
tukey_result = pairwise_tukeyhsd(data['value'], data['group'])

# Print the results
print(tukey_result)
```

## [Bonferroni Correction](#bonferroni-correction)
- **Description**: Adjusts p-values when performing multiple comparisons to reduce the chances of obtaining false-positive results (Type I errors). The correction divides the significance level by the number of comparisons.
- **Assumptions**:
  - Multiple hypothesis tests are performed.
  - Significance level is adjusted to account for the number of tests.
- **Null Hypothesis**: Each individual hypothesis test has its own null hypothesis (e.g., no significant difference between groups).

### R Code
```r
# Example p-values from multiple hypothesis tests
p_values <- c(value_1, ...)  # Replace with actual p-values from your tests

# Number of tests performed
number_of_tests <- length(p_values)

# Apply Bonferroni correction to adjust p-values
adjusted_p_values <- p.adjust(p_values, method = "bonferroni")

# Display the adjusted p-values
adjusted_p_values
```

### Python Code
```python
import numpy as np
from statsmodels.stats.multitest import multipletests

# Example p-values from multiple hypothesis tests
p_values = np.array([value_1, ...])  # Replace with actual p-values from your tests

# Apply Bonferroni correction to adjust p-values
adjusted_results = multipletests(p_values, alpha=0.05, method='bonferroni')

# Extract the corrected p-values
corrected_p_values = adjusted_results[1]

# Print the adjusted p-values
print(corrected_p_values)
```

## [Levene's Test](#levenes-test)
- **Description**: Tests for the equality of variances across multiple groups. Commonly used to check the assumption of equal variances in ANOVA or T-tests.
- **Assumptions**:
  - Data is independent.
  - Does not assume normality of the data.
- **Null Hypothesis**: The variances of the groups are equal.

### R Code
```r
# Assume data in 'data' and group labels in 'group'
library(car)
leveneTest(data ~ group)
```

### Python Code
```python
from scipy import stats

# Sample data
group1 = [...] # Replace with your data
group2 = [...] # Replace with your data

# Perform Levene's test
w_stat, p_value = stats.levene(group1, group2)

# Print the results
print(f"Levene Statistic: {w_stat}, P-value: {p_value}")
```

## [Testing of Normality Distribution for Paired Test](#testing-of-normality-distribution-for-paired-test)
- **Description**: Used to check if the differences between paired samples are normally distributed, an assumption for parametric tests like the paired t-test.
- **Assumptions**:
  - Data is dependent (paired samples).
  - The distribution of the differences is approximately normal.
- **Null Hypothesis**: The differences between the pairs are normally distributed.

### R Code
```r
# Assume data in 'before' and 'after'
diff <- before - after
qqnorm(diff)
qqline(diff)
shapiro.test(diff) # null hypothesis: data is normally distributed
```

### Python Code
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Sample data
before = [...] # Replace with your data
after = [...] # Replace with your data

# Compute differences
diff = np.array(before) - np.array(after)

# Q-Q plot
stats.probplot(diff, dist="norm", plot=plt)
plt.show()

# Shapiro-Wilk test for normality
w_stat, p_value = stats.shapiro(diff)

# Print the results
print(f"Shapiro-Wilk Statistic: {w_stat}, P-value: {p_value}")
```

## [Decision tree](#decision-tree)
[↑ Back to Top](#statistical-tests)


[![](https://mermaid.ink/img/pako:eNqlVttS4zgQ_RWVqdl9IVUBdmprvDNsKTdgIOROGGQetLYSq1AkI8tDUlPz79uWHMcJIcXO8kCS7nPafTkt64cXqoh5vvfhAxpzI5iPRoYanhoeUoHGLDVoxAQLDVcStTida7oIZCDhM4nRzTCQwCztoaBp2mIzlFD4zYzmIZpxIfyj1qePZ390jlOj1RPzj87OzorvtRcemdg_TZbHoRJK-0f1ev2vSiip5Kto9fqfp43Wr0SL2IxmwvyHOHrpn9Tzz5X73IqL4G_9H5NL9YIWVK7Qd6o5_Uew9O9HVKudowY5KW2PgLTGJjndAEtri5yjbXsgG9bTJoOMCp6P53sexVk7uVWajTmQbeu4ID3JaolWidJ2egaG-bj2XpJmzGvpc0Y1Q3OlIsnSFKkZNMaskYHsWOyVjZTSRSIYyCOLmDS_p8jULM73_c181pSvVcqUi1AtNxkEsmlB19WCYDIJk3noaqcc8Garxv3IQF5bbJc0uKR6hYSaWxkjzeYaioMW7ObqGLekC4rgUi04SP4dtEDeWGLPEfMSBZeM6oOcluX0YejPW2N09oGzVwfpHEOQTtWBfkMnOxEC2bfQEcSYa5UlKZopjRgN42qHHGhs9VWBUYMEo7DnSrLtho4sYVJVSj7BXCVcrocQ5tixRd6RpgpjTSWIY7AZ9sA6p6RftmO3OQ5xT26rq265U-v5RvrQ3RQUFCqtmaBmT4fvLfSBjBLALmwSkMrTFiUPObQ4jCEfrlmEnEjzRSs8DXIlNxLbuGFD3Yri5oFa1pjWnmqwkzNuk90tspNIthPaDez0gDuk3KeUzyWLarbMst3YnQv44lCSBeZyX5IXzne1P8nTyuzNm7leuhhfSZdKWZvG3Ei2QpNNlk4x-Jq0lwm8XaDsmWbPGeiJsxSdf0EfS1nhm_2gzxYDD3NrjLvvECp2m4tvSYenMdNQG1vSsHLkFV3ulbtUbiLuV1bHPbnnHINDrS4ww32tdiuJR4f4BWa8j-8WB09IOz8RUKKSzCndLTLU_Ho0BeeOTCR7Nwt4E8eb_h9Z3LkY92TKRBj_QgSIUWzptzelhd25hR_sO-iFrmDcvTv8KhkHa-Aymb0wADoZNhrkWmfpExW1KRWCp5Unuhk1mmQIBdBcqAs4UjN4F7zx7ILQIh3NWQSnlQvmHXsLBocXj-BK9iO_WASeidmCBZ4PX4vLS-AF8idAaWbUaCVDzzc6Y8delkTw7OIy5vkzKlKwJlQ-KLX5zSJulO66a5-9_R17IOl5XCB-_guHqHJm?type=png)](https://mermaid.live/edit#pako:eNqlVttS4zgQ_RWVqdl9IVUBdmprvDNsKTdgIOROGGQetLYSq1AkI8tDUlPz79uWHMcJIcXO8kCS7nPafTkt64cXqoh5vvfhAxpzI5iPRoYanhoeUoHGLDVoxAQLDVcStTida7oIZCDhM4nRzTCQwCztoaBp2mIzlFD4zYzmIZpxIfyj1qePZ390jlOj1RPzj87OzorvtRcemdg_TZbHoRJK-0f1ev2vSiip5Kto9fqfp43Wr0SL2IxmwvyHOHrpn9Tzz5X73IqL4G_9H5NL9YIWVK7Qd6o5_Uew9O9HVKudowY5KW2PgLTGJjndAEtri5yjbXsgG9bTJoOMCp6P53sexVk7uVWajTmQbeu4ID3JaolWidJ2egaG-bj2XpJmzGvpc0Y1Q3OlIsnSFKkZNMaskYHsWOyVjZTSRSIYyCOLmDS_p8jULM73_c181pSvVcqUi1AtNxkEsmlB19WCYDIJk3noaqcc8Garxv3IQF5bbJc0uKR6hYSaWxkjzeYaioMW7ObqGLekC4rgUi04SP4dtEDeWGLPEfMSBZeM6oOcluX0YejPW2N09oGzVwfpHEOQTtWBfkMnOxEC2bfQEcSYa5UlKZopjRgN42qHHGhs9VWBUYMEo7DnSrLtho4sYVJVSj7BXCVcrocQ5tixRd6RpgpjTSWIY7AZ9sA6p6RftmO3OQ5xT26rq265U-v5RvrQ3RQUFCqtmaBmT4fvLfSBjBLALmwSkMrTFiUPObQ4jCEfrlmEnEjzRSs8DXIlNxLbuGFD3Yri5oFa1pjWnmqwkzNuk90tspNIthPaDez0gDuk3KeUzyWLarbMst3YnQv44lCSBeZyX5IXzne1P8nTyuzNm7leuhhfSZdKWZvG3Ei2QpNNlk4x-Jq0lwm8XaDsmWbPGeiJsxSdf0EfS1nhm_2gzxYDD3NrjLvvECp2m4tvSYenMdNQG1vSsHLkFV3ulbtUbiLuV1bHPbnnHINDrS4ww32tdiuJR4f4BWa8j-8WB09IOz8RUKKSzCndLTLU_Ho0BeeOTCR7Nwt4E8eb_h9Z3LkY92TKRBj_QgSIUWzptzelhd25hR_sO-iFrmDcvTv8KhkHa-Aymb0wADoZNhrkWmfpExW1KRWCp5Unuhk1mmQIBdBcqAs4UjN4F7zx7ILQIh3NWQSnlQvmHXsLBocXj-BK9iO_WASeidmCBZ4PX4vLS-AF8idAaWbUaCVDzzc6Y8delkTw7OIy5vkzKlKwJlQ-KLX5zSJulO66a5-9_R17IOl5XCB-_guHqHJm)