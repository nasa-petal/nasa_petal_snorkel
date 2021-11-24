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
import sys, wget, pickle, json
sys.path.insert(0,'../snorkel')
from snorkel.labeling.model import LabelModel
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import MajorityLabelVoter
from ast import literal_eval
from utils import smaller_models
from create_labeling_functions import create_labeling_functions
from tqdm import trange
import pandas as pd 
import os.path as osp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from create_labeling_functions import create_labeling_functions
import numpy as np 
#import csv file and load train/test/split of dataset

golden_json_url = 'https://raw.githubusercontent.com/nasa-petal/data-collection-and-prep/main/golden.json'

filename = 'golden.json'
large_model = 'large_model_trained.pickle'
small_models = 'small_models_trained.pickle'
if not osp.exists(filename):
    wget.download(golden_json_url)

with open(filename,'r') as f:
    golden_json = json.load(f)

'''
    Train snorkel using Golden.json 
'''
datalist = list()
for paper in golden_json:
    data = dict()
    data['text'] = ' '.join(literal_eval(paper['title']) + literal_eval(paper['abstract']))
    data['doi'] = paper['doi']
    data['paperid'] = paper['paper']
    datalist.append(data)
df = pd.DataFrame(datalist)

# loop through all Golden JSON and Predict 
if not osp.exists('golden_lf.pickle'):
    df_bio = pd.read_csv(r'./biomimicry_functions_enumerated.csv')
    labels = dict(zip(df_bio['function_enumerated'].tolist(),df_bio['function'].tolist()))

    labeling_function_list = create_labeling_functions(r'./biomimicry_functions_enumerated.csv', r'./biomimicry_function_rules.csv')
    applier = PandasLFApplier(lfs=labeling_function_list)
    L_golden = applier.apply(df=df)

    labels_overlap, L_matches, translators, translators_to_str, L_match_all, global_translator,global_translator_str, dfs = smaller_models(L_golden,5,2,labels_list=labels,df=df)

    with open('golden_lf.pickle','wb') as f:
        pickle.dump({'L_golden':L_golden,'labels_overlap':labels_overlap,'L_matches':L_matches, 
                    'translators':translators,'translators_to_str':translators_to_str, 
                    'L_matches_all':L_match_all, 'global_translator':global_translator, 
                    'global_translator_str':global_translator_str,'dfs':dfs},f)

with open('golden_lf.pickle','rb') as f:
    data = pickle.load(f) 
    L_golden = data['L_golden']
    print("Unique Matches in golden.json: ")
    print(*np.unique(L_golden).tolist(), sep = ", ")
    L_matches = data['L_matches']
    labels_overlap = data['labels_overlap']
    translators = data['translators']
    translators_to_str = data['translators_to_str']
    dfs = data['dfs']
    global_translator = data['global_translator']
    L_match_all = data['L_matches_all']
'''
    Loop to evaluate all the smaller models
    Note: some models are very small to splitting them into test and train can be tricky 
'''
models = list()
for i in trange(len(L_matches),desc="training small models"):
    L_match = L_matches[i]
    # TODO split the dataset so theres an equal amount of all labels
    L_train = L_match
    L_test = L_match 

    labels = labels_overlap[i]
    cardinality = len(labels)   # How many labels to predict 
    majority_model = MajorityLabelVoter(cardinality=cardinality)
    preds_train = majority_model.predict(L=L_train)                 # Looks at each text and sees which label is predicted the most 
    
    # Train LabelModel - this outputs probabilistic floats 
    label_model = LabelModel(cardinality=cardinality, verbose=True, device = 'cpu')
    label_model.fit(L_train=L_train, n_epochs=350, log_freq=50, seed=123)
    probs_train = label_model.predict_proba(L=L_train)  # This gives you the probability of which label paper falls under 

    models.append(label_model) # this label model can help predict the type of paper

with open(small_models,'wb') as f:
    pickle.dump({"Label_models":models, 'labels_overlap':labels_overlap,
        'translators':translators,'translators_to_str':translators_to_str,
        'texts_df':dfs},f)

# Training a single large model
cardinality = len(global_translator)
majority_model = MajorityLabelVoter(cardinality=cardinality)
preds_train = majority_model.predict(L=L_match_all)
label_model = LabelModel(cardinality=cardinality, verbose=True, device = 'cpu')
label_model.fit(L_train=L_match_all, n_epochs=300, log_freq=50, seed=123)

with open(large_model,'wb') as f:
    pickle.dump({"Label_model":label_model,'global_translator':global_translator,
        'text_df':df},f)
