This is an overview of the markdown contents of all the notebooks / scripts in this directory.
# 00_benchmark_clinician_suspicion


This notebook benchmarks how well clinical suspicion does for predicting IAI-I. It only uses data from PECARN.

## compare rule to clinician suspicion

## counts should match pecarn paper

# 01_match_features_psrc_pecarn


**load the data**

## see what feats matched / are missing

## look for feat shift

**continuous features**

**categorical features**

**binary feats**

**missing features**

**write csvs**

## how are vars related to outcome

# 02_benchmark_pecarn_rule


## check bivariable associations

## recreate / evaluate rule

**look at pecarn patients**

**look at psrc patients**

## look at errors for subgroups
**here we consider the riskier subgroups (young age, female)**

# 03_eda


## combined

**correlations between features**

**individual correlations with outcome**

**subgroup risks (with sizes)**

**intersectional subgroup risks (with sizes)**

**joint correlations (or risks) with outcome joint**

## features scatter plots

#### continuous features

**we can cut GCSScore as whether it is 15 or not (14 is already pretty bad)**

#### scatter plots

# 04_sample_splitting


# 05_fit_interpretable_models


## fit simple models

**decision tree**

**bayesian rule list**

**rulefit**

**slim - sparse linear integer model**

**skope rules**

**greedy (CART) rule list**

**irf - iterative Random Forest**

**rf**

## look at all the results

# 06_benchmark_new_rule


**test individual rule**

# 07_measure_predictive_limits


## look at different buckets

## fit simple models

**decision tree**

