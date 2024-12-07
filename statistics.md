# Quick Guide to Statistical Tests

---

### Introduction

This guide provides a quick overview of how to do common statistical tests in R and Python. Each section shows you the code and explains how to understand the results. It's important to make sure your data meets the requirements (assumptions) of each test to get accurate results. In general, we compare the p-value from the test to a significance level (usually 0.05). If the p-value is smaller than the significance level, we reject the null hypothesis (meaning we think there's a real effect).

## Statistical Tests

### Descriptive Statistics
*   *Purpose: Provides an overview of data characteristics and summarizes data sets.*
*   **[Descriptive Statistics](#descriptive-statistics-1)**

### Comparing Means
*   *Purpose: Tests comparing the means of one or more groups to determine significant differences.*

#### Statistical Tests for Comparing Group Means
*   ##### Between Two Groups
    *   **[One Sample T-test](#one-sample-t-test)**
    *   **[Two Sample T-test](#two-sample-t-test)**
    *   **[Paired Sample T-test](#paired-sample-t-test)**
    *   **[Paired Z-test](#paired-z-test)**
*   ##### Between More Than Two Groups
    *   **[One-Way ANOVA](#one-way-anova)**
    *   **[Two-Way ANOVA](#two-way-anova)**
*   ##### Non-Parametric Tests for Comparing Means
    *   **[Kruskal-Wallis H-test](#kruskal-wallis-h-test)**
    *   **[Friedman Test](#friedman-test)**

### Comparing Distributions
*   *Purpose: Non-parametric tests comparing distributions of two or more groups.*
*   **[Wilcoxon Signed Rank Test](#wilcoxon-signed-rank-test)**
*   **[Mann-Whitney U Test](#mann-whitney-u-test)**

### Categorical Data Analysis
*   *Purpose: Tests for independence and associations between categorical variables.*
*   **[Chi-Square Test for Independence](#chi-square-test-for-independence)**
*   **[Fisher's Exact Test](#fishers-exact-test-detailed-use-case)**

### *Post hoc* Multiple Comparisons
*   *Purpose: Adjusts for multiple comparisons after ANOVA to control Type I error rates.*
*   **[Tukey's Honestly Significant Difference (HSD) Test](#tukeys-honestly-significant-difference-hsd-test)**
*   **[Bonferroni Correction](#bonferroni-correction)**

### Assumption Checks
*   *Purpose*: Tests to check assumptions such as normality and equal variances required for parametric tests.
*   **[Levene's Test](#levenes-test)**
*   **[Testing of Normality Distribution for Paired Test](#testing-of-normality-distribution-for-paired-test)**

### Choosing the Right Statistical Test
*   *Purpose*: Overview of decision tree analysis methods and their applications.
*   **[Decision tree](#decision-tree)**

## [Decision Tree](#decision-tree) 

---

## Descriptive Statistics: Summarizing Your Data

*   **Purpose:** Gives you a basic overview of your data, showing things like averages and how spread out the data is.
*   **Includes:** Mean (average), median (middle value), standard deviation (how much data varies around the mean), and range (difference between the highest and lowest values).

#### R Code

```r
# Load necessary libraries
library(dplyr)
library(tibble)

# Example data
data <- data.frame(
  group = c("A", "A", "A", "B", "B", "C", "C", "C"),
  value = c(12, 15, 18, 22, 25, 10, 13, 16)
)

# Descriptive statistics
summary_stats <- data %>%
  group_by(group) %>%
  summarise(
    mean = mean(value, na.rm = TRUE),  # 'value' is your data column
    median = median(value, na.rm = TRUE),
    sd = sd(value, na.rm = TRUE),
    min = min(value, na.rm = TRUE),
    max = max(value, na.rm = TRUE),
    n = n() # Number of samples
  )

# Print the results
print(summary_stats)
```

#### Python Code (for pandas DataFrame)

```python
import pandas as pd

# Example data
data = pd.DataFrame({
    'group': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C'],
    'value': [12, 15, 18, 22, 25, 10, 13, 16]
})

# Remove rows with NaN (missing) values in 'value' column
data_clean = data.dropna(subset=['value'])

# Descriptive statistics
summary_stats = data_clean.groupby('group').agg(
    mean=('value', 'mean'),
    median=('value', 'median'),
    sd=('value', 'std'),
    min=('value', 'min'),
    max=('value', 'max'),
    n=('value', 'size') # Number of samples
).reset_index()

print(summary_stats)
```

[↑ Back to Top](#statistical-tests)

## Tests for Comparing Means (Averages)

*   **Purpose:** These tests help you figure out if the averages of two or more groups are different from each other in a meaningful way.

### Between Two Groups

#### One Sample T-test

*   **Description:** Checks if the average of one group is different from a specific number (a known or hypothesized population average). You don't need to know how spread out the whole population's data is.
*   **Assumptions:**
    *   Data points should be independent.
    *   Data should be continuous.
    *   Ideally, sample size is less than 30, but test can work with larger samples.
    *   The sample means should be approximately normally distributed.
*   **Null Hypothesis:** The average of your group is the same as the specific number you're comparing it to.

##### R Code

```r
# Sample data
sample_data <- c(48, 52, 55, 49, 51, 53, 50, 54)

# Known population mean (the specific number you're comparing to)
population_mean <- 50 

# Perform the one-sample t-test
t_test_result <- t.test(sample_data, mu = population_mean)

# Display the results
print(t_test_result)
```

##### Python Code

```python
from scipy import stats

# Sample data
sample_data = [48, 52, 55, 49, 51, 53, 50, 54]

# Known population mean
population_mean = 50

# Perform the one-sample t-test
t_stat, p_value = stats.ttest_1samp(sample_data, population_mean)

# Print the results
print(f"T-statistic: {t_stat}, P-value: {p_value}")
```

#### Z-test

*   **Description:** Checks if the average of a sample is different from a known or hypothesized population average when you know how spread out the whole population's data is. Usually used with large samples (more than 30).
*   **Assumptions:**
    *   Data points should be independent.
    *   Data should be continuous.
    *   Ideally, sample size is more than 30.
    *   You know the population's standard deviation (how spread out the data is).
*   **Null Hypothesis:** The average of your sample is the same as the known population average.

##### R Code

```r
library(BSDA) # Install if needed: install.packages("BSDA")

# Sample data
sample_data <- c(52, 55, 58, 53, 56, 60, 54, 57, 59, 55, 58, 56, 57, 54, 56, 58, 55, 59, 56, 57, 55, 58, 54, 56, 57, 59, 56, 58, 55, 57)

# Known population mean and standard deviation
population_mean <- 55
population_sd <- 2

# Perform the Z-test
z_test_result <- z.test(x = sample_data, mu = population_mean, sigma.x = population_sd)

# Display the results
print(z_test_result)
```

##### Python Code

```python
from statsmodels.stats.weightstats import ztest

# Sample data
sample_data = [52, 55, 58, 53, 56, 60, 54, 57, 59, 55, 58, 56, 57, 54, 56, 58, 55, 59, 56, 57, 55, 58, 54, 56, 57, 59, 56, 58, 55, 57]

# Known population mean and standard deviation
population_mean = 55
population_sd = 2

# Perform the Z-test
z_stat, p_value = ztest(sample_data, value=population_mean)

# Print the results
print(f"Z-statistic: {z_stat}, P-value: {p_value}")
```

[↑ Back to Top](#statistical-tests)

#### Two Sample T-test

*   **Description:** Checks if the averages of two separate (independent) groups are different from each other. You don't need to know how spread out the whole population's data is.
*   **Assumptions:**
    *   Data points should be independent within and between groups.
    *   Data should be continuous.
    *   Ideally, sample size in each group is less than 30, but can work with larger samples.
    *   The sample means should be approximately normally distributed.
    *   Assumes the two groups have roughly the same spread (variance) in their data - you can check this with Levene's test.
*   **Null Hypothesis:** The averages of the two groups are the same.

##### R Code

```r
# Sample data for two groups
group1 <- c(32, 35, 38, 33, 36, 40, 34, 37)
group2 <- c(28, 31, 34, 29, 32, 35, 30, 33)

# Perform the two-sample t-test
t_test_result <- t.test(group1, group2, var.equal = TRUE) # Use var.equal = FALSE if variances are unequal

# Display the results
print(t_test_result)
```

##### Python Code

```python
from scipy import stats

# Sample data for two groups
group1 = [32, 35, 38, 33, 36, 40, 34, 37]
group2 = [28, 31, 34, 29, 32, 35, 30, 33]

# Perform the two-sample t-test
t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=True) # Use equal_var=False if variances are unequal

# Print the results
print(f"T-statistic: {t_stat}, P-value: {p_value}")
```

[↑ Back to Top](#statistical-tests)

#### Paired Sample T-test

*   **Description:** Checks if the averages of two related groups (like "before" and "after" measurements on the same people) are different. You don't need to know how spread out the whole population's data is.
*   **Assumptions:**
    *   Data should be continuous.
    *   The two measurements are taken from the same subjects or matched pairs.
    *   Ideally, sample size (number of pairs) is less than 30, but can work with larger samples.
    *   The differences between the pairs should be approximately normally distributed.
*   **Null Hypothesis:** The average difference between the paired measurements is zero (no change).

##### R Code

```r
# Paired data
before <- c(22, 25, 28, 21, 24, 26, 23, 27)
after <- c(25, 27, 30, 24, 26, 29, 25, 29)

# Perform the paired t-test
paired_t_test_result <- t.test(before, after, paired = TRUE)

# Display the results
print(paired_t_test_result)
```

##### Python Code

```python
from scipy import stats

# Paired sample data
before = [22, 25, 28, 21, 24, 26, 23, 27]
after = [25, 27, 30, 24, 26, 29, 25, 29]

# Perform the paired t-test
t_stat, p_value = stats.ttest_rel(before, after)

# Print the results
print(f"T-statistic: {t_stat}, P-value: {p_value}")
```

[↑ Back to Top](#statistical-tests)

##### Paired Z-test

*   **Description:** Checks if the averages of two related groups (like "before" and "after" measurements on the same people) are different when you know how spread out the whole population's data is. Usually used when you have lots of data (more than 30 pairs).
*   **Assumptions:**
    *   Data should be continuous.
    *   The two measurements are taken from the same subjects or matched pairs.
    *   You know the population standard deviation of the differences between pairs.
    *   The differences between the pairs should be approximately normally distributed.
*   **Null Hypothesis:** The average difference between the paired measurements is zero (no change).

##### R Code

```r
# Paired data
before <- c(55, 58, 60, 53, 56, 62, 57, 59, 54, 56, 58, 61, 55, 57, 59, 56, 58, 60, 54, 57, 56, 59, 58, 60, 55, 57, 59, 56, 58, 60)
after <- c(57, 60, 63, 55, 59, 64, 59, 62, 56, 58, 61, 63, 57, 60, 62, 58, 60, 63, 56, 59, 58, 62, 60, 63, 57, 60, 62, 58, 61, 62)

# Calculate the mean difference
mean_diff <- mean(after - before)

# Calculate the standard error of the mean difference
sd_diff <- 3 # known population standard deviation of the differences
n <- length(before)

if (n <= 30) warning("Sample size is not large (n > 30). Consider using a paired t-test if population standard deviation is unknown.")

std_error <- sd_diff / sqrt(n)

# Compute the Z statistic
z_stat <- mean_diff / std_error

# Compute the p-value (two-tailed test)
p_value <- 2 * pnorm(-abs(z_stat))

# Output results
list(Z_Statistic = z_stat, P_Value = p_value)
```

##### Python Code

```python
import numpy as np
from scipy import stats

# Paired sample data
before = [55, 58, 60, 53, 56, 62, 57, 59, 54, 56, 58, 61, 55, 57, 59, 56, 58, 60, 54, 57, 56, 59, 58, 60, 55, 57, 59, 56, 58, 60]
after = [57, 60, 63, 55, 59, 64, 59, 62, 56, 58, 61, 63, 57, 60, 62, 58, 60, 63, 56, 59, 58, 62, 60, 63, 57, 60, 62, 58, 61, 62]
sd_diff = 3  # Known population standard deviation of the differences

# Calculate mean difference
mean_diff = np.mean(np.array(after) - np.array(before))

# Calculate the standard error
n = len(before)

if (n <= 30):
  print("Warning: Sample size is not large (n > 30). Consider using a paired t-test if the population standard deviation is unknown.")

std_error = sd_diff / np.sqrt(n)

# Compute the Z statistic
z_stat = mean_diff / std_error

# Compute the p-value (two-tailed test)
p_value = 2 * stats.norm.cdf(-abs(z_stat))

# Print the results
print(f"Z-Statistic: {z_stat}, P-value: {p_value}")
```

[↑ Back to Top](#statistical-tests)

#### Between More Than Two Groups

##### One-Way ANOVA

*   **Description:** Checks if the averages of three or more separate (independent) groups are different from each other.
*   **Assumptions:**
    *   Data points should be independent within and between groups.
    *   Data should be continuous.
    *   Data within each group should be approximately normally distributed.
    *   The groups should have roughly the same spread (variance) in their data - you can check this with Levene's test.
*   **Null Hypothesis:** The averages of all the groups are the same.

###### R Code

```r
# Data for three groups
group1 <- c(12, 15, 18, 13, 16, 19, 14, 17)
group2 <- c(20, 23, 25, 21, 24, 26, 22, 25)
group3 <- c(8, 10, 12, 9, 11, 13, 10, 12)

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

###### Python Code

```python
from scipy import stats
import pandas as pd

# Data for three groups
group1 = [12, 15, 18, 13, 16, 19, 14, 17]
group2 = [20, 23, 25, 21, 24, 26, 22, 25]
group3 = [8, 10, 12, 9, 11, 13, 10, 12]

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

##### Kruskal-Wallis H-test

*   **Description:** Checks if the distributions of three or more separate groups are different. This is a good alternative to ANOVA when your data doesn't meet ANOVA's requirements (like normality).
*   **Assumptions:**
    *   Data points should be independent within and between groups.
    *   Data should be at least ordinal (can be ranked).
    *   The groups should have roughly the same shape of distribution.
*   **Null Hypothesis:** All the groups have the same distribution of data.

###### R Code

```r
# Data for three groups
group1 <- c(5, 7, 9, 6, 8, 10, 7, 9)
group2 <- c(12, 15, 17, 13, 16, 18, 14, 17)
group3 <- c(3, 5, 7, 4, 6, 8, 5, 7)

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

###### Python Code

```python
from scipy import stats
import pandas as pd

# Data for three groups
group1 = [5, 7, 9, 6, 8, 10, 7, 9]
group2 = [12, 15, 17, 13, 16, 18, 14, 17]
group3 = [3, 5, 7, 4, 6, 8, 5, 7]

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

[↑ Back to Top](#statistical-tests)

##### Two-Way ANOVA

*   **Description:** Used when you have two factors (like "treatment type" and "gender") that you think might affect the outcome you're measuring. It checks if each factor has an effect, and also if the factors interact with each other (like if a treatment works differently for men and women).
*   **Assumptions:**
    *   Data points should be independent.
    *   Data should be continuous.
    *   The spread of the data (variance) should be roughly the same across all groups.
    *   The residuals (the differences between the observed values and the values predicted by the model) should be approximately normally distributed.
*   **Null Hypothesis:** Neither factor has an effect on the outcome, and there's no interaction between the factors.

###### R Code

```r
# Load necessary library
library(dplyr)
library(ggplot2)

# Example data
data <- data.frame(
  factor1 = factor(rep(c("Level1", "Level2"), each = 15)),
  factor2 = factor(rep(c("A", "B", "C"), times = 10)),
  response = c(10, 12, 11, 13, 14, 12, 13, 14, 15, 13, 14, 15, 16, 14, 15, 9, 11, 10, 12, 13, 11, 12, 13, 14, 12, 13, 14, 15, 13, 14)
)

# Perform Two-Way ANOVA
anova_results <- aov(response ~ factor1 * factor2, data = data)

# Summary of the ANOVA
summary(anova_results)
```

###### Python Code

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Example data
data = pd.DataFrame({
    'factor1': np.repeat(['Level1', 'Level2'], 15),
    'factor2': np.tile(['A', 'B', 'C'], 10),
    'response': [10, 12, 11, 13, 14, 12, 13, 14, 15, 13, 14, 15, 16, 14, 15, 9, 11, 10, 12, 13, 11, 12, 13, 14, 12, 13, 14, 15, 13, 14]
})

# Perform Two-Way ANOVA
model = ols('response ~ C(factor1) * C(factor2)', data=data).fit()
anova_results = sm.stats.anova_lm(model, typ=2)

# Summary of the ANOVA
print(anova_results)
```

[↑ Back to Top](#statistical-tests)

##### Friedman Test

*   **Description:** Used to compare three or more related groups (like multiple measurements on the same people). It's a good alternative to a repeated-measures ANOVA when your data doesn't meet ANOVA's requirements.
*   **Assumptions:**
    *   You have one group of subjects measured multiple times (or under different conditions).
    *   Data can be ranked (at least ordinal).
*   **Null Hypothesis:** There are no differences between the groups (treatments or conditions).

###### R Code

```r
# Example of Friedman's Test in R

# Load necessary library
library(dplyr)

# Example data
data <- data.frame(
  subject = factor(rep(1:10, each = 3)),
  treatment = factor(rep(c("A", "B", "C"), times = 10)),
  response = c(7, 8, 6, 5, 6, 4, 7, 9, 8, 6, 7, 5, 4, 5, 3, 8, 9, 7, 9, 10, 8, 7, 8, 6, 5, 6, 4, 8, 9, 7)
)

# Perform Friedman's Test
friedman_test <- friedman.test(response ~ treatment | subject, data = data)

# Print results
print(friedman_test)
```

###### Python Code

```python
# Example of Friedman's Test in Python
import pandas as pd
from scipy.stats import friedmanchisquare

# Example data
data = pd.DataFrame({
    'subject': list(range(1, 11)) * 3,
    'treatment': ['A'] * 10 + ['B'] * 10 + ['C'] * 10,
    'response': [7, 8, 6, 5, 6, 4, 7, 9, 8, 6, 7, 5, 4, 5, 3, 8, 9, 7, 9, 10, 8, 7, 8, 6, 5, 6, 4, 8, 9, 7]
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

[↑ Back to Top](#statistical-tests)

### Non-Parametric Tests for Comparing Distributions

*   **Purpose:** These tests are used to compare groups when you can't assume your data is normally distributed (bell-shaped).

#### Wilcoxon Signed Rank Test

*   **Description:** Used to compare two related groups (like "before" and "after" measurements on the same people) when you can't assume the differences between the groups are normally distributed.
*   **Assumptions:**
    *   Data should be at least ordinal (can be ranked).
    *   The two measurements are taken from the same subjects or matched pairs.
    *   The differences between pairs are independent.
    *   The differences between pairs are symmetric around the median.
*   **Null Hypothesis:** The median difference between the paired measurements is zero (no change).

##### R Code

```r
# Paired data
before <- c(15, 18, 20, 12, 16, 19, 14, 17)
after <- c(17, 20, 23, 14, 18, 22, 16, 19)

# Perform the Wilcoxon signed-rank test
wilcoxon_result <- wilcox.test(before, after, paired = TRUE)

# Display the results
print(wilcoxon_result)
```

##### Python Code

```python
from scipy import stats

# Paired sample data
before = [15, 18, 20, 12, 16, 19, 14, 17]
after = [17, 20, 23, 14, 18, 22, 16, 19]

# Perform the Wilcoxon signed-rank test
wilcoxon_result = stats.wilcoxon(before, after)

# Print the results
print(f"Test statistic: {wilcoxon_result.statistic}, P-value: {wilcoxon_result.pvalue}")
```

[↑ Back to Top](#statistical-tests)

#### Mann-Whitney U Test

*   **Description:** Used to compare two separate (independent) groups when you can't assume the data is normally distributed.
*   **Assumptions:**
    *   Data points should be independent within and between groups.
    *   Data should be at least ordinal (can be ranked).
    *   The two groups should have roughly the same shape of distribution.
*   **Null Hypothesis:** The distributions of the two groups are the same.

##### R Code

```r
# Sample data for two groups
group1 <- c(8, 10, 12, 9, 11, 13, 10, 12)
group2 <- c(5, 7, 9, 6, 8, 10, 7, 9)

# Perform the Mann-Whitney U test
mann_whitney_result <- wilcox.test(group1, group2)

# Display the results
print(mann_whitney_result)
```

##### Python Code

```python
from scipy import stats

# Sample data for two groups
group1 = [8, 10, 12, 9, 11, 13, 10, 12]
group2 = [5, 7, 9, 6, 8, 10, 7, 9]

# Perform the Mann-Whitney U test
mann_whitney_result = stats.mannwhitneyu(group1, group2)

# Print the results
print(f"U-statistic: {mann_whitney_result.statistic}, P-value: {mann_whitney_result.pvalue}")
```

[↑ Back to Top](#statistical-tests)

### Analyzing Categorical Data: Independence and Association

*   **Purpose:** These tests deal with data that falls into categories (like "yes/no" or "low/medium/high") rather than being measured on a continuous scale.

#### Chi-Square Test for Independence

*   **Description:** Checks if two categorical variables are related or independent. For example, is there a relationship between gender and voting preference?
*   **Assumptions:**
    *   The variables are categorical.
    *   All observations are independent.
    *   You need a decent amount of data in each category (usually at least 5 in each cell of the contingency table).
*   **Null Hypothesis:** The two variables are independent (not related).

##### R Code

```r
# Example data
data <- matrix(c(25, 15, 10, 30), nrow = 2, dimnames = list(c("Male", "Female"), c("Voted", "Did Not Vote")))

# Perform the Chi-Square Test
chi_square_result <- chisq.test(data)

# Display the results
print(chi_square_result)
```

##### Python Code

```python
from scipy import stats
import numpy as np

# Example data
data = np.array([[25, 15], [10, 30]])

# Perform the Chi-Square Test
chi2_stat, p_value, dof, expected = stats.chi2_contingency(data)

# Print the results
print(f"Chi-square statistic: {chi2_stat}, P-value: {p_value}")
```

[↑ Back to Top](#statistical-tests)

#### Fisher's Exact Test

*   **Description:** Similar to the Chi-Square test, but it's used when you have small amounts of data in some categories, especially in a 2x2 table.
*   **Assumptions:**
    *   The variables are categorical.
    *   All observations are independent.
*   **Null Hypothesis:** The two variables are independent (not related).

##### R Code

```r
# Example data
data <- matrix(c(4, 1, 2, 3), nrow = 2, dimnames = list(c("Group A", "Group B"), c("Success", "Failure")))

# Perform Fisher's Exact Test
fisher_result <- fisher.test(data)

# Display the results
print(fisher_result)
```

##### Python Code

```python
from scipy import stats
import numpy as np

# Example data
data = np.array([[4, 1], [2, 3]])

# Perform Fisher's Exact Test
fisher_result = stats.fisher_exact(data)

# Print the results
print(f"Odds ratio: {fisher_result[0]}, P-value: {fisher_result[1]}")
```

[↑ Back to Top](#statistical-tests)

### Post Hoc Tests: Multiple Comparisons After ANOVA

*   **Purpose:** If you do an ANOVA and find that there are differences between your groups, these tests help you figure out exactly which groups are different from each other. They adjust for the fact that you're doing multiple comparisons.

#### Tukey's Honestly Significant Difference (HSD) Test

*   **Description:** A common post hoc test used after ANOVA to compare all possible pairs of group averages.
*   **Assumptions:**
    *   You've already done an ANOVA and found a significant result.
    *   The spread of the data (variance) should be roughly the same across all groups.
*   **Null Hypothesis:** The averages of each pair of groups being compared are the same.

##### R Code

```r
# Example data
data <- data.frame(
  value = c(12, 15, 18, 13, 16, 19, 14, 17, 20, 23, 25, 21, 24, 26, 22, 25, 8, 10, 12, 9, 11, 13, 10, 12),
  group = factor(rep(c("Group1", "Group2", "Group3"), each = 8))
)

# Perform One-Way ANOVA (if you haven't already)
anova_result <- aov(value ~ group, data = data)

# Perform Tukey's HSD test
tukey_result <- TukeyHSD(anova_result)

# Display the results
print(tukey_result)
```

##### Python Code

```python
```python
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd

# Example data
data = pd.DataFrame({
    'value': [12, 15, 18, 13, 16, 19, 14, 17, 20, 23, 25, 21, 24, 26, 22, 25, 8, 10, 12, 9, 11, 13, 10, 12],
    'group': ['Group1']*8 + ['Group2']*8 + ['Group3']*8
})

# Perform One-Way ANOVA (if you haven't already)
anova_model = sm.formula.ols('value ~ group', data=data).fit()
anova_result = sm.stats.anova_lm(anova_model, typ=2)

# Perform Tukey's HSD test
tukey_result = pairwise_tukeyhsd(data['value'], data['group'])

# Print the results
print(tukey_result)
```

[↑ Back to Top](#statistical-tests)

#### Bonferroni Correction

*   **Description:** A simple way to adjust for multiple comparisons. It makes it harder to find a significant result, to reduce the chance of false positives.
*   **Assumptions:**
    *   You're doing multiple hypothesis tests.
*   **Null Hypothesis:** Each individual hypothesis test has its own null hypothesis (e.g., no significant difference between groups).

##### R Code

```r
# Example p-values from multiple hypothesis tests
p_values <- c(0.04, 0.02, 0.01, 0.08, 0.15)

# Number of tests performed
number_of_tests <- length(p_values)

# Apply Bonferroni correction to adjust p-values
adjusted_p_values <- p.adjust(p_values, method = "bonferroni")

# Display the adjusted p-values
adjusted_p_values
```

##### Python Code

```python
import numpy as np
from statsmodels.stats.multitest import multipletests

# Example p-values from multiple hypothesis tests
p_values = np.array([0.04, 0.02, 0.01, 0.08, 0.15])

# Apply Bonferroni correction to adjust p-values
adjusted_results = multipletests(p_values, alpha=0.05, method='bonferroni')

# Extract the corrected p-values
corrected_p_values = adjusted_results[1]

# Print the adjusted p-values
print(corrected_p_values)
```

[↑ Back to Top](#statistical-tests)

### Checking Statistical Test Assumptions

*   **Purpose:** Before you rely on the results of a statistical test, it's important to check if your data meets the requirements (assumptions) of that test.

#### Levene's Test

*   **Description:** Checks if the spread of the data (variance) is roughly the same across different groups. This is important for tests like ANOVA and t-tests.
*   **Assumptions:**
    *   The samples from the populations are independent.
*   **Null Hypothesis:** The variances of the groups are equal.

##### R Code

```r
# Example data
data <- data.frame(
  value = c(12, 15, 18, 13, 16, 19, 14, 17, 20, 23, 25, 21, 24, 26, 22, 25, 8, 10, 12, 9, 11, 13, 10, 12),
  group = factor(rep(c("Group1", "Group2", "Group3"), each = 8))
)

# Perform Levene's test
library(car)
leveneTest(value ~ group, data = data)
```

##### Python Code

```python
from scipy import stats

# Sample data
group1 = [12, 15, 18, 13, 16, 19, 14, 17]
group2 = [20, 23, 25, 21, 24, 26, 22, 25]
group3 = [8, 10, 12, 9, 11, 13, 10, 12]

# Perform Levene's test
w_stat, p_value = stats.levene(group1, group2, group3)

# Print the results
print(f"Levene Statistic: {w_stat}, P-value: {p_value}")
```

[↑ Back to Top](#statistical-tests)

#### Testing of Normality Distribution for Paired Test

*   **Description:** Used to check if the differences between paired samples are approximately normally distributed (bell-shaped). This is important for tests like the paired t-test.
*   **Assumptions:**
    *   The two measurements are taken from the same subjects or matched pairs.
    *   The distribution of the differences should be approximately normal.
*   **Null Hypothesis:** The differences between the pairs are normally distributed.

##### R Code

```r
# Example data
before <- c(22, 25, 28, 21, 24, 26, 23, 27)
after <- c(25, 27, 30, 24, 26, 29, 25, 29)

# Calculate the differences
diff <- before - after

# Create a Q-Q plot
qqnorm(diff)
qqline(diff)

# Perform the Shapiro-Wilk test
shapiro.test(diff)
```

##### Python Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Example data
before = [22, 25, 28, 21, 24, 26, 23, 27]
after = [25, 27, 30, 24, 26, 29, 25, 29]

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

[↑ Back to Top](#statistical-tests)

### Decision tree
[![](https://mermaid.ink/img/pako:eNqdVm1z4jYQ_isaZ6b9AhnjHDnitndjQwh5IZCXlruY-6DaMmjOSI4kX-A8-e-VLQNSsNNOGWbs8e7z7O6jXUm5FdIIWa61YDBdgsfBnMwJkL8wgZwPUAxSyOAKCYZDEOMkcY_CMO7CsMUFo9-Re3RyclK9t19wJJauk65bIU0oc49s2_7tDR2h5IAxgqgX_2_GCMUwS0TFFffisxi-z8XWbscunhv1NLgVuxeM6AtYQbIBPyDD8O8E8c_fXNfdRmu3PwE_7-ysrxWuNPRzZw8zLIN8Pv8ETKuy-6X9PL_LYIIFFPgH-vyqm4aFiQjNpqzn0gq-Il46XQQTgtopoyllAlMCBOKiyNvQXQPe0hI3CvpL3ObPGWQILCiNCOIc0FiqKpo4FMtQD39ZhudwlSYIPIgsQkT8yoFobyne5jDUcrjSwTOchHTdXICC90vgtS6a7IcUkSLwTuStjsr7xtCxwV0BrvXaxoGPCWQbkNAF5kJ2L0MLJmWSMteVdq2VdhuMZdNgQlcYJv-FQFHc6PEniqMQJ8EEQfYv8W-0-NPcp2IJnvXCKQMrvEbRvtxpgSgdDYUKgrvg4Ps3DTQuiErH-6Bjon8B5Ydkj1K4u9J9lk93aW-XSVm-5Lf6iu-znOmifA2mUgku-ySkjKEEigY1vmhqPAUPqUTJ6Za9ySD5_hZc22v3JdbzZMaYyWpVo_Jt1pXZzy_JvqU0n2oX8PTsvX5N-cqlStUbNMng9Q2m8-DtuIFYrnBq5FonjDfQww2D3eBxvCAoapf6vDuEnm9kclFXk68HGTXWdGEwXdbX5AB8KHFtaSM96lUwhoS0Z0ssCNqAP98vaqBAE7mTLxjN0t06V4ZptZEr476CiVHBXZ0WEz2r-0YtpgbTQx3TVGd6PGQqnCrXh8J1RFeUhyiCxe6DxUYBn8p99wVugHc7-curVVLBkUDskMD3ghlKwqVcpUaCiuZRr8n3g2uW8e8wac9gkmD-3mGloFWtfj-_lw0AhWztMYI8Y_qM-cZk-INgyDCK5LS_x69AW_5zY4prQhiHrn8hu6Tm3K58jab2R0bjlA0NBUhkCAEoQQeHloJvE7tUkWquAf7I6Jfr_HydorBQKGboOUMkxNImY_8BujtuY0Bu6iG_a_6XRilXweMLbW6cauGMQ9Qb6xeNcqTlJWM_zyFqWn_jLPRugyHmS8Rkz6E1DBvuKFbLWiG5z-NI3nDzgmhuiSVaobnlytfqLje35uRVusJM0IcNCS1XsAy1rCyNZIcNMJR345XlxjDh8msKyROlq60TirCgbKwu0eVdunSx3NxaW_J6eXbs2J1T-XfOTj98_NhpWRv5uWsf245tn_VOu51e1zlxXlvWz5LVPu51P9hdp9OTIMc57UmE7JTFskrg9R9SDrj8?type=png)](https://mermaid.live/edit#pako:eNqdVm1z4jYQ_isaZ6b9AhnjHDnitndjQwh5IZCXlruY-6DaMmjOSI4kX-A8-e-VLQNSsNNOGWbs8e7z7O6jXUm5FdIIWa61YDBdgsfBnMwJkL8wgZwPUAxSyOAKCYZDEOMkcY_CMO7CsMUFo9-Re3RyclK9t19wJJauk65bIU0oc49s2_7tDR2h5IAxgqgX_2_GCMUwS0TFFffisxi-z8XWbscunhv1NLgVuxeM6AtYQbIBPyDD8O8E8c_fXNfdRmu3PwE_7-ysrxWuNPRzZw8zLIN8Pv8ETKuy-6X9PL_LYIIFFPgH-vyqm4aFiQjNpqzn0gq-Il46XQQTgtopoyllAlMCBOKiyNvQXQPe0hI3CvpL3ObPGWQILCiNCOIc0FiqKpo4FMtQD39ZhudwlSYIPIgsQkT8yoFobyne5jDUcrjSwTOchHTdXICC90vgtS6a7IcUkSLwTuStjsr7xtCxwV0BrvXaxoGPCWQbkNAF5kJ2L0MLJmWSMteVdq2VdhuMZdNgQlcYJv-FQFHc6PEniqMQJ8EEQfYv8W-0-NPcp2IJnvXCKQMrvEbRvtxpgSgdDYUKgrvg4Ps3DTQuiErH-6Bjon8B5Ydkj1K4u9J9lk93aW-XSVm-5Lf6iu-znOmifA2mUgku-ySkjKEEigY1vmhqPAUPqUTJ6Za9ySD5_hZc22v3JdbzZMaYyWpVo_Jt1pXZzy_JvqU0n2oX8PTsvX5N-cqlStUbNMng9Q2m8-DtuIFYrnBq5FonjDfQww2D3eBxvCAoapf6vDuEnm9kclFXk68HGTXWdGEwXdbX5AB8KHFtaSM96lUwhoS0Z0ssCNqAP98vaqBAE7mTLxjN0t06V4ZptZEr476CiVHBXZ0WEz2r-0YtpgbTQx3TVGd6PGQqnCrXh8J1RFeUhyiCxe6DxUYBn8p99wVugHc7-curVVLBkUDskMD3ghlKwqVcpUaCiuZRr8n3g2uW8e8wac9gkmD-3mGloFWtfj-_lw0AhWztMYI8Y_qM-cZk-INgyDCK5LS_x69AW_5zY4prQhiHrn8hu6Tm3K58jab2R0bjlA0NBUhkCAEoQQeHloJvE7tUkWquAf7I6Jfr_HydorBQKGboOUMkxNImY_8BujtuY0Bu6iG_a_6XRilXweMLbW6cauGMQ9Qb6xeNcqTlJWM_zyFqWn_jLPRugyHmS8Rkz6E1DBvuKFbLWiG5z-NI3nDzgmhuiSVaobnlytfqLje35uRVusJM0IcNCS1XsAy1rCyNZIcNMJR345XlxjDh8msKyROlq60TirCgbKwu0eVdunSx3NxaW_J6eXbs2J1T-XfOTj98_NhpWRv5uWsf245tn_VOu51e1zlxXlvWz5LVPu51P9hdp9OTIMc57UmE7JTFskrg9R9SDrj8)