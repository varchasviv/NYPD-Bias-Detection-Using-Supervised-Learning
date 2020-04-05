# NYPD-Bias-Detection-using-Supervised-Learning
DSU/DIPD Data Competition files for Team 322

# Quick Look
An analysis of the role of demographic variables in the context of arrests by the NYPD. 
We use Random Forest & Naive Bayes Classifiers.

TL;DR We use Level Specific Models as an implicit measure of discriminatory bias. (models capturing structure within data)

We developed a 'bias-test' methodology of generating subsets of the data based on the levels of one demographic factor and then 'hiding' the chosen variable during modeling. This results in level specific models based on a single demographic variable that are then tested against the data subsets originating from the other levels of the demographic variable. If we observe a clear misclassification skewed to specific levels of the categorical target variable, then we could conclude that discrimination is at play, and also infer the nature of the bias. While if level-specific models perform well on data subsets originating from other levels, then we can conclude that there is no clear discriminatory bias.


# File Guide

To read our final report, refer to *CORRECT VERSION -Data Comp - Team 322.ipynb* or *Data Comp - Team 322.pdf*

# Contributors:
- Avijit Singh Nalwa (@AvijitNalwa)
- Varchasvi Vedula (@varchasviv) 
