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
import sys
sys.path.insert(0,'../snorkel')
from snorkel.labeling.model import LabelModel
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling import filter_unlabeled_dataframe
from snorkel.utils import probs_to_preds
import pickle
from utils import smaller_models
from create_labeling_functions import create_labeling_functions
from tqdm import trange
import pandas as pd 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#import csv file and load train/test/split of dataset
from utils import load_dataset

df = load_dataset()
#get labeling functions (lfs)

labeling_function_list = create_labeling_functions(r'./biomimicry_functions_enumerated.csv', r'./biomimicry_function_rules.csv')
df_bio = pd.read_csv(r'./biomimicry_functions_enumerated.csv')
labels = dict(zip(df_bio['function_enumerated'].tolist(),df_bio['function'].tolist()))

applier = PandasLFApplier(lfs=labeling_function_list)
L_match = applier.apply(df=df)
labels_overlap, L_matches, translators, translators_to_str, L_match_all, global_translator, dfs = smaller_models(L_match,5,2,labels_list=labels,df=df)


'''
    Loop to evaluate all the smaller models
    Note: some models are very small to splitting them into test and train can be tricky 
'''
models = list()
for i in trange(len(L_matches)):
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
    label_model.fit(L_train=L_train, n_epochs=300, log_freq=50, seed=123)
    probs_train = label_model.predict_proba(L=L_train)  # This gives you the probability of which label paper falls under 

    models.append(label_model) # this label model can help predict the type of paper

with open('small_models_trained.pickle','wb') as f:
    pickle.dump({"Label_models":models, 'labels_overlap':labels_overlap,
        'translators':translators,'translators_to_str':translators_to_str,
        'texts_df':dfs},f)

# Training a single large model 
cardinality = len(global_translator)
majority_model = MajorityLabelVoter(cardinality=cardinality)
preds_train = majority_model.predict(L=L_match_all)
label_model = LabelModel(cardinality=cardinality, verbose=True, device = 'cpu')
label_model.fit(L_train=L_match_all, n_epochs=300, log_freq=50, seed=123)

with open('large_model_trained.pickle','wb') as f:
    pickle.dump({"Label_model":label_model,'global_translator':global_translator,'translators_to_str':translators_to_str,
        'texts_df':df},f)
