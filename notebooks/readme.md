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

## look at errors on splits

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


Fit interpretable models to the training set and test on validation sets. Uses imodels package as of 10/25/2020 (v0.2.5).

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

# 06_permutation_importances


Fit interpretable models to the training set and test on validation sets. Uses imodels package as of 10/25/2020 (v0.2.5).

## look at importances for all the models

# 07_benchmark_compare_new_rules


**test individual rule**

## test overlap of different rules

**rules make different predictions**

**difference in predictions makes little difference for IAI-I**

# 08_measure_predictive_limits


## look at different buckets

## fit simple models

**decision tree**

# pecarn_predict


