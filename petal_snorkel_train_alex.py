'''
 -*- coding: utf-8 -*-
    Title: petal_snorkel_train_alex.py
    Description: Trains snorkel on Alex's csv papers and predicts at the end. 
               Paht thinks that this is how snorkel works. Trying to use another model on a different dataset ends badly.
 
    Authors: paht.juangphanich@nasa.gov
'''
from genericpath import exists
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import os.path as osp
from tqdm import trange
import sys
sys.path.insert(0, '../snorkel')
from create_labeling_functions import create_labeling_functions
from utils import smaller_models, single_model_to_dict, compare_single_model_dicts, normalize_L, compare_single_model_dict, load_dataset
from copy import deepcopy
from ast import literal_eval
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel
import wget
import pickle
import json


'''
    Load Alex's dataset 
'''
df = load_dataset()
labeling_function_list = create_labeling_functions(r'./biomimicry_functions_enumerated.csv', r'./biomimicry_function_rules.csv')
df_bio = pd.read_csv(r'./biomimicry_functions_enumerated.csv')
labels = dict(zip(df_bio['function_enumerated'].tolist(),df_bio['function'].tolist()))

# TODO: Need to make sure L_matches is normalized using global_json predictions 

applier = PandasLFApplier(lfs=labeling_function_list)
L_match = applier.apply(df=df)

'''
    Loop through all Alex Dataset and create L-matrix
'''
labels_overlap, L_matches, translators, translators_to_str, L_match_all, global_translator, global_translator_str, dfs = smaller_models(L_match, 5, 2, labels_list=labels, df=df)

'''
    Train small models
    Note: some models are very small to splitting them into test and train can be tricky 
'''
models = list()
for i in trange(len(L_matches), desc="training small models"):
    L_match = L_matches[i]
    # TODO split the dataset so theres an equal amount of all labels
    L_train = L_match
    L_test = L_match

    labels = labels_overlap[i]
    cardinality = len(labels)   # How many labels to predict
    majority_model = MajorityLabelVoter(cardinality=cardinality)
    # Looks at each text and sees which label is predicted the most
    preds_train = majority_model.predict(L=L_train)

    # Train LabelModel - this outputs probabilistic floats
    label_model = LabelModel(
        cardinality=cardinality, verbose=True, device='cpu')
    label_model.fit(L_train=L_train, n_epochs=350, log_freq=100, seed=123)
    # This gives you the probability of which label paper falls under
    probs_train = label_model.predict_proba(L=L_train)

    # this label model can help predict the type of paper
    models.append(label_model)

'''
    Train large models
    Note: some models are very small to splitting them into test and train can be tricky 
'''

# Training a single large model
cardinality = len(global_translator)
majority_model = MajorityLabelVoter(cardinality=cardinality)
preds_train = majority_model.predict(L=L_match_all)
label_model = LabelModel(cardinality=cardinality, verbose=True, device='cpu')
label_model.fit(L_train=L_match_all, n_epochs=300, log_freq=50, seed=123)

'''
    Evaluation using smaller models
'''
smaller_model_L = list()
for i in trange(len(labels_overlap)):            
    smaller_model_L.append(normalize_L(L=L_match[i],translator=translator[i]))

results = list()
for i in range(len(smaller_model_data['Label_models'])):
    results.extend(single_model_to_dict(L_matches[i],smaller_model_data['Label_models'][i], smaller_model_data['translators_to_str'][i],i,dfs[i]))
# Filter papers by unique doi 

df_sm = pd.DataFrame(results)
doi_all = df_sm['doi'].unique()
results = list()
for doi in doi_all:
    df_unique_doi = df_sm[df_sm['doi']==doi]
    papers = [df_unique_doi.iloc[i].to_dict() for i in range(len(df_unique_doi))]
    for p in range(len(papers)): 
        if p == 0:
            results.append(papers[p])
        else:
            results[-1] = compare_single_model_dict(results[-1],papers[p])
df_sm = pd.DataFrame(results)
df_sm.to_csv("golden json matches small models.csv")


'''
    Evaluate using larger model
'''
if osp.exists(large_model):
    with open(large_model,'rb') as f:
        large_model_data = pickle.load(f)
        large_label_model = large_model_data['Label_model']
        global_translator = large_model_data['global_translator'] # old labels to new 
        global_translator_str = large_model_data['global_translator_str']

        large_model_L = normalize_L(L=L_golden,translator=global_translator)

large_model_results = single_model_to_dict(large_model_L,large_label_model, global_translator_str,0,df)
df_lg = pd.DataFrame(large_model_results)
df_lg.to_csv("golden json matches large model.csv")