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
from utils import load_dataset, evaluate_model,normalize_L
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

def single_model_to_dict(label_model:LabelModel, translator_to_str:Dict[int,str],L_match:np.ndarray,model_index:int) -> List[Dict]:
    """Evaluates the prediction accuracy of a single model

    Args:
        label_model (LabelModel): Model
        translator_to_str (Dict[int,str]): dictionary containing keys -> string conversion
        L_match (np.ndarray): this is the numpy array from pandas lf filter
        model_index (int): index of model 

    Returns:
        List[Dict]: List of results
    """
    model = label_model
    translator = translator_to_str
    results_for_each_paper = list()
    temp = evaluate_model(L_match, model, translator,model_index)
    for p, paper in enumerate(temp):
        probabilities = [t['probability'] for t in paper]
        labels = [t['label'] for t in paper]
        model_indicies = [t['model_index'] for t in paper]
                
        zipped = list(zip(probabilities,labels,model_indicies))
        zipped = sorted(zipped, reverse=True)
        probabilities, labels, model_indicies = zip(*zipped)
    
        results_for_each_paper.append({ 
                'title':df.iloc[p]['title'], 'abstract':df.iloc[p]['abstract'],
                'doi':df.iloc[p]['doi']})
        if not pd.isna(df.iloc[p]['label_level_1']):
            results_for_each_paper[p]['label'] = ast.literal_eval(df.iloc[p]['label_level_1'])
        else:
            results_for_each_paper[p]['label'] = 'not found'

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

def compare_single_model_dicts(result1:List[Dict], result2:List[Dict]) -> List[Dict]: 
    """Compares one result with another and outputs the best case

    Args:
        result1 (List[Dict]): [description]
        result2 (List[Dict]): [description]

    Returns:
        List[Dict]: [description]
    """
    for i in range(len(result1)):

        if result2[i]['probability-snorkel-1'] > result1[i]['probability-snorkel-1']: 
            result1[i]['label-snorkel-1']  = result2[i]['label-snorkel-1']
            result1[i]['label-snorkel-1']  = result2[i]['label-snorkel-1']
        if result2[i]['probability-snorkel-2'] > result1[i]['probability-snorkel-2']: 
            result1[i]['label-snorkel-2']  = result2[i]['label-snorkel-2']
            result1[i]['label-snorkel-2']  = result2[i]['label-snorkel-2']
        if result2[i]['probability-snorkel-3'] > result1[i]['probability-snorkel-3']: 
            result1[i]['label-snorkel-3']  = result2[i]['label-snorkel-3']
            result1[i]['label-snorkel-3']  = result2[i]['label-snorkel-3']
    return result1

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