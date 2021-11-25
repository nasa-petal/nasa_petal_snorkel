'''
    Predicts what label something is based on trained data from golden json. 
    This prediction doesn't really work at all. I think snorkel needs to be used on what it is trained on. 
    If you have a new dataset then you should add it to the existing and run the predictions after training. 

    This example shows how to use the consensus label model approach to predict the "label" of any given text using snorkel

'''
from copy import deepcopy
import sys
from types import LambdaType
from typing import Dict, List
sys.path.insert(0,'../snorkel')
from snorkel.labeling.model import LabelModel
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import MajorityLabelVoter
from utils import smaller_models
import wget
import os.path as osp 
import pickle, json
import pandas as pd 
from tqdm import trange
import numpy as np 
from ast import literal_eval
from utils import load_dataset,normalize_L, single_model_to_dict, compare_single_model_dicts
from create_labeling_functions import create_labeling_functions
import ast
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


large_model = 'large_model_trained.pickle'
small_models = 'small_models_trained.pickle'

if osp.exists(small_models):
    with open(small_models,'rb') as f:
        smaller_model_data = pickle.load(f)
        smaller_model_L = list()
        for i in trange(len(smaller_model_data['labels_overlap'])):            
            labels = smaller_model_data['labels_overlap'][i]
            translator = smaller_model_data['translators'][i]
            translator_to_str = smaller_model_data['translators_to_str'][i]
            smaller_model_L.append(normalize_L(L=L_match,translator=translator))
            

if osp.exists(large_model):
    with open(large_model,'rb') as f:
        large_model_data = pickle.load(f)
        large_label_model = large_model_data['Label_model']
        global_translator = large_model_data['global_translator'] # old labels to new 
        global_translator_str = large_model_data['global_translator_str']

        large_model_L = normalize_L(L=L_match,translator=global_translator)

'''
    Evaluation using smaller models
'''
best_results = None
for i in range(len(smaller_model_data['Label_models'])):
    results = single_model_to_dict(smaller_model_data['Label_models'][i], smaller_model_data['translators_to_str'][i],smaller_model_L[i],i)
    if i ==0:
        best_results = deepcopy(results)
    else:
        best_results = compare_single_model_dicts(best_results,results)
df_sm = pd.DataFrame(best_results)
df_sm.to_csv("alex paper matches small models.csv")
'''
    Evaluate using larger model
'''
large_model_results = single_model_to_dict(large_label_model, global_translator_str,large_model_L,0)
df_lg = pd.DataFrame(large_model_results)
df_lg.to_csv("alex paper matches large model.csv")