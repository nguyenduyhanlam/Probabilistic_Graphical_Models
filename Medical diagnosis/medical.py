# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 15:16:55 2020

@author: User
"""
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import ParameterEstimator
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import BayesianEstimator

data = pd.read_csv('test.csv')
data = data.fillna('Unknown')
data_head = data.head()
print(data_head)

data_columns = data.columns

def CreateLinks(columns):
    links = []
    
    for i in range(1, len(columns)):
        couple = (columns[i], columns[0])
        links.append(couple)
    
    return links

links = CreateLinks(data_columns)
model = BayesianModel(links)

pe = ParameterEstimator(model, data)

# Print ParameterEstimator unconditional
pe_symptom1 = pe.state_counts('Symptom_1')
print(pe_symptom1)

# Print ParameterEstimator conditional disease
pe_disease = pe.state_counts('Disease')
print(pe_disease)

mle = MaximumLikelihoodEstimator(model, data)

# Print MaximumLikelihoodEstimator unconditional
mle_symptom1 = mle.estimate_cpd('Symptom_1')
print(mle_symptom1)

# Print MaximumLikelihoodEstimator conditional
#mle_disease = mle.estimate_cpd('Disease')
#print(mle_disease)

# Calibrate all CPDs of `model` using MLE:
model.fit(data, estimator=MaximumLikelihoodEstimator)

est = BayesianEstimator(model, data)
est_disease = est.estimate_cpd('Disease', prior_type='BDeu', equivalent_sample_size=10)
print(est_disease)