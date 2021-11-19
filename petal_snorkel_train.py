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
labels_overlap, L_matches, translators, translators_to_str, L_match, global_translator,dfs = smaller_models(L_match,5,2,labels_list=labels,df=df)

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
    
    # *Note: This part can be replaced by scibert
    # df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(X=dfs[i], y=probs_train, L=L_match)
    # vectorizer = CountVectorizer(ngram_range=(1, 5))
    # X_train = vectorizer.fit_transform(df_train_filtered.text.tolist())
    # # X_test = vectorizer.transform(df_test.text.tolist())

    # preds_train_filtered = probs_to_preds(probs=probs_train_filtered)
    # sklearn_model = LogisticRegression(C=1e3, solver="liblinear")
    # sklearn_model.fit(X=X_train, y=preds_train_filtered)


    # majority_acc = majority_model.score(L=L_test, Y=Y_test, tie_break_policy="random")["accuracy"]
    # label_model_acc = label_model.score(L=L_test, Y=Y_test, tie_break_policy="random")["accuracy"]

'''
    Loop to evaluate the larger model
'''

#     df = LFAnalysis(L=L_train, lfs=labeling_function_list).lf_summary()
#     with open('lf_analysis.pickle','wb') as f:
#         pickle.dump({"lf_analysis":df, 'L_train':L_train,'L_test':L_test},f)

#     # Br
# if os.path.exists('lf_analysis.pickle'):
#     with open('lf_analysis.pickle','rb') as f:
#         data = pickle.load(f)
#         lf_analysis = data['lf_analysis']
#         L_train = data['L_train']
#         L_test = data['L_test']

# majority_model = MajorityLabelVoter(cardinality=98)
# preds_train = majority_model.predict(L=L_train)

# label_model = LabelModel(cardinality=98, verbose=True, device = 'cpu')
# label_model.fit(L_train=L_train, n_epochs=1000, log_freq=100, seed=123)

# LFAnalysis(L=L_train, lfs=labeling_function_list).lf_summary()

# df_train_filtered, preds_train_filtered = filter_unlabeled_dataframe(
#     X=df_train, y=preds_train, L=L_train)

# df_train["label"] = label_model.predict(L=L_train, tie_break_policy="abstain")

# label_model.save("snorkel_model.pkl")

# df_train.to_csv("results.csv")
