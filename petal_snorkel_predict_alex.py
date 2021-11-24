'''
    Predicts what label something is based on 
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
from utils import load_dataset, evaluate_model
from create_labeling_functions import create_labeling_functions
import ast
'''
    Load Alex's dataset 
'''
df = load_dataset()
labeling_function_list = create_labeling_functions(r'./biomimicry_functions_enumerated.csv', r'./biomimicry_function_rules.csv')
df_bio = pd.read_csv(r'./biomimicry_functions_enumerated.csv')
labels = dict(zip(df_bio['function_enumerated'].tolist(),df_bio['function'].tolist()))

applier = PandasLFApplier(lfs=labeling_function_list)
L_match = applier.apply(df=df)
labels_overlap, L_matches, translators, translators_to_str, L_match_all, global_translator, dfs = smaller_models(L_match,5,2,labels_list=labels,df=df)

large_model = 'large_model_trained.pickle'
small_models = 'small_models_trained.pickle'

if osp.exists(small_models):
    with open(small_models,'rb') as f:
        smaller_model_data = pickle.load(f)
        all_labels = dict()
        for i in trange(len(smaller_model_data)):
            translator = smaller_model_data['translators_to_str'][i]
            for k,v in translator.items():
                if v not in all_labels.keys():
                    all_labels[v] = list() 

if osp.exists(large_model):
    with open(large_model,'rb') as f:
        large_model_data = pickle.load(f)

def evaluate_models(label_models:List, translators_to_str:List):
    """Evaluates small models using Alex dataset 

    Args:
        label_models (List): List of label models from snorkel 
        translators_to_str (List): list of dictionaries translating matches to strings 

    Returns:
        [type]: [description]
    """
    # Create a copy of the all labels dictionary for each paper
    results_for_each_paper = list()
    for p in trange(L_match.shape[0]):
        L = L_match[p,:].reshape(1,-1)
        results_for_each_paper.append({ 
                'title':df.iloc[p]['title'], 'abstract':df.iloc[p]['abstract'],
                'doi':df.iloc[p]['doi']})
        if not pd.isna(df.iloc[p]['label_level_1']):
            results_for_each_paper[p]['label'] = ast.literal_eval(df.iloc[p]['label_level_1'])
        else:
            results_for_each_paper[p]['label'] = 'not found'

        results_for_each_paper[p]['label-snorkel-1'] = ''
        results_for_each_paper[p]['label-snorkel-2'] = ''
        results_for_each_paper[p]['label-snorkel-3'] = ''
        results_for_each_paper[p]['probability-snorkel-1'] = 0
        results_for_each_paper[p]['probability-snorkel-2'] = 0
        results_for_each_paper[p]['probability-snorkel-3'] = 0
        results_for_each_paper[p]['model-index-snorkel-1'] = 0
        results_for_each_paper[p]['model-index-snorkel-2'] = 0
        results_for_each_paper[p]['model-index-snorkel-3'] = 0
        labels = list()
        probabilities = list() 
        model_indicies = list() 
        for i in range(len(label_models)):
            model = label_models[i]
            translator = translators_to_str[i]
            temp = evaluate_model(L, model,translator,i)
            temp = temp[0]
            temp_probabilities = [t['probability'] for t in temp]
            temp_labels = [t['label'] for t in temp]
            temp_model_indicies = [t['model_index'] for t in temp]
            labels.extend(temp_labels)
            probabilities.extend(temp_probabilities)
            model_indicies.extend(temp_model_indicies)
        
        zipped = list(zip(probabilities,labels,model_indicies))
        zipped = sorted(zipped, reverse=True)
        probabilities, labels, model_indicies = zip(*zipped)

        results_for_each_paper[p]['label-snorkel-1'] = labels[0]
        results_for_each_paper[p]['label-snorkel-2'] = labels[1]
        results_for_each_paper[p]['label-snorkel-3'] = labels[2]
        results_for_each_paper[p]['probability-snorkel-1'] = probabilities[0]
        results_for_each_paper[p]['probability-snorkel-2'] = probabilities[1]
        results_for_each_paper[p]['probability-snorkel-3'] = probabilities[2]
        results_for_each_paper[p]['model-index-snorkel-1'] = model_indicies[0]
        results_for_each_paper[p]['model-index-snorkel-2'] = model_indicies[1]
        results_for_each_paper[p]['model-index-snorkel-3'] = model_indicies[2]
    return results_for_each_paper

'''
    Evaluation using smaller models
'''
# small_model_results = evaluate_models(smaller_model_data['Label_models'], smaller_model_data['translators_to_str'])
# df_sm = pd.DataFrame(small_model_results)
# df_sm.to_csv("alex paper matches div-conquer.csv")

'''
    Evaluate using larger model
'''
labels = {-1:"no_match", **labels}
large_model_results = evaluate_models([large_model_data['Label_model']], [labels])
df_lg = pd.DataFrame(large_model_results)
df_lg.to_csv("alex paper matches large modeldiv-conquer.csv")