Demonstrating the use of PDR/PCS in emergency medicine on the PECARN dataset.


# PCS documentation


1. Domain problem formulation (narrative). Clearly state the real-world question and describe prior work related to this question. Indicate how this question can be answered in the context of a model or analysis.

Want to be able to identify the risk of clinically important traumatic brain injury (ciTBI) among children. This information can be used to triage CT imaging.

2. Data collection and storage (narrative). Describe how the data were generated, including experimental design principles, and reasons why data is relevant to answer the domain question. Describe where data is stored and how it can be accessed by others.

Protocol for screening subjects is given in the paper (i.e. children presenting within 24 h of non-trivial head trauma). Data is now open-source and available as a series of csv and accompanying pdf files providing details on how it was collected.

3. Data cleaning and preprocessing (narrative, code, visualization). Describe steps taken to convert raw data into data used for analysis, and why these preprocessing steps are justified. Ask whether more than one preprocessing methods should be used and examine their impacts on the final data results.

The definition of the outcome is the most difficult part. Categorical features are on-hot encoded.

4. Exploratory data analysis (narrative, code, visualization). Describe any preliminary analyses that influenced modeling decisions or conclusions along with code and visualizations to support these decisions.

Split up preverbal (<2 years of age) and verbal (>=2 years of age) patients.

5. Modeling and Post-hoc analysis (narrative, code, visualization). Carry out PCS inference in the context of the domain question. Specify appropriate model and data perturbations. If necessary, specify null hypotheses and associated perturbations.

sfdg

6. Interpretation of results (narrative and visualization). Translate the data results to draw conclusions and/or make recommendations in the context of domain problem.

asd


# Reference

- docs generated with pdoc3
- stability-selection is taken from [this repo](https://github.com/scikit-learn-contrib/stability-selection) with minor adjustments
    - to use, cd to lib/stability-selection then run `python setup.py install develop`

