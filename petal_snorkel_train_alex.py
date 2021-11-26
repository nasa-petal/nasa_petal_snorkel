# -*- coding: utf-8 -*-
# ------------------------------------------------- #
# Title: petal_snorkel.py
# Description: main file for running snorkel ML model
# ChangeLog: (Name, Date: MM-DD-YY, Changes)
# <ARalevski, 10-01-2021, created script>
# <PahtJ, 10-15-21, updated pickle>
# <ARalevski, 10-17-2021, updated cardinality>
# Authors: alexandra.ralevski@gmail.com, paht.juangphanich@nasa.gov
# ------------------------------------------------- #
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
from utils import smaller_models, single_model_to_dict, compare_single_model_dicts, normalize_L, compare_single_model_dict
from copy import deepcopy
from ast import literal_eval
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel
import wget
import pickle
import json

# import csv file and load train/test/split of dataset

golden_json_url = 'https://raw.githubusercontent.com/nasa-petal/data-collection-and-prep/main/golden.json'

filename = 'golden.json'
large_model = 'large_model_trained.pickle'
small_models = 'small_models_trained.pickle'
'''
    Download golden json 
'''
if not osp.exists(filename):
    wget.download(golden_json_url)

with open(filename, 'r') as f:
    golden_json = json.load(f)

'''
    Train snorkel using Golden.json
    Loads golden_json to a dataframe.
'''
datalist = list()
for paper in golden_json:
    data = dict()
    data['text'] = ' '.join(literal_eval(
        paper['title']) + literal_eval(paper['abstract']))
    data['doi'] = paper['doi']
    data['paperid'] = paper['paper']
    data['title'] = ' '.join(literal_eval(paper['title']))
    data['abstract'] = ' '.join(literal_eval(paper['abstract']))
    data['label_level_1'] = paper['level1'] # Assign this because it's coming from golden json 
    datalist.append(data)
df = pd.DataFrame(datalist)

df_bio = pd.read_csv(r'./biomimicry_functions_enumerated.csv')
labels = dict(
    zip(df_bio['function_enumerated'].tolist(), df_bio['function'].tolist()))



'''
    Loop through all Golden JSON and create L-matrix and smaller models.
'''
if not osp.exists('golden_lf.pickle'):
    labeling_function_list = create_labeling_functions(r'./biomimicry_functions_enumerated.csv', r'./biomimicry_function_rules.csv')
    applier = PandasLFApplier(lfs=labeling_function_list)
    L_golden = applier.apply(df=df)
    labels_overlap, L_matches, translators, translators_to_str, L_match_all, global_translator, global_translator_str, dfs = smaller_models(
    L_golden, 5, 2, labels_list=labels, df=df)
    with open('golden_lf.pickle', 'wb') as f:
        pickle.dump({'L_golden': L_golden, 'labels_overlap': labels_overlap, 'L_matches': L_matches,
                    'translators': translators, 'translators_to_str': translators_to_str,
                     'L_matches_all': L_match_all, 'global_translator': global_translator,
                     'global_translator_str': global_translator_str, 'dfs': dfs}, f)

with open('golden_lf.pickle', 'rb') as f:
    data = pickle.load(f)
    L_golden = data['L_golden']
    print("Unique Matches in golden.json: ")
    print(*np.unique(L_golden).tolist(), sep=", ")
    L_matches = data['L_matches']
    labels_overlap = data['labels_overlap']
    translators = data['translators']
    translators_to_str = data['translators_to_str']
    global_translator_str = data['global_translator_str']
    dfs = data['dfs']
    global_translator = data['global_translator']
    L_match_all = data['L_matches_all']

'''
    Train small models
    Note: some models are very small to splitting them into test and train can be tricky 
        Each small model is trained on it's own set of papers. 
'''
if not osp.exists(small_models):
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
            cardinality=cardinality, verbose=True, device='cuda') # smaller models can be trained with GPU 
        label_model.fit(L_train=L_train, n_epochs=350, log_freq=100, seed=123)
        # This gives you the probability of which label paper falls under
        probs_train = label_model.predict_proba(L=L_train)

        # this label model can help predict the type of paper
        models.append(label_model)

    with open(small_models, 'wb') as f:
        pickle.dump({"Label_models": models, 'labels_overlap': labels_overlap,
                     'translators': translators, 'translators_to_str': translators_to_str,
                     'texts_df': dfs}, f)

'''
    Train large models
    Note: some models are very small to splitting them into test and train can be tricky 
'''
if not osp.exists(large_model):
    # Training a single large model
    cardinality = len(global_translator)
    majority_model = MajorityLabelVoter(cardinality=cardinality)
    preds_train = majority_model.predict(L=L_match_all)
    label_model = LabelModel(cardinality=cardinality, verbose=True, device='cpu')
    label_model.fit(L_train=L_match_all, n_epochs=300, log_freq=50, seed=123)

    with open(large_model, 'wb') as f:  # Saves the large model 
        pickle.dump({"Label_model": label_model, 'global_translator': global_translator,
                    'global_translator_str': global_translator_str, 'text_df': df}, f)


'''
    Evaluation using smaller models
'''
if osp.exists(small_models):
    with open(small_models,'rb') as f:
        smaller_model_data = pickle.load(f)
        smaller_model_L = list()
        for i in trange(len(smaller_model_data['labels_overlap'])):            
            labels = smaller_model_data['labels_overlap'][i]
            translator = smaller_model_data['translators'][i]
            translator_to_str = smaller_model_data['translators_to_str'][i]
            smaller_model_L.append(normalize_L(L=L_golden,translator=translator))

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