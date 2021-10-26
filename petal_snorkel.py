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

from snorkel.labeling.model import LabelModel
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling import filter_unlabeled_dataframe
import pickle, os

#import csv file and load train/test/split of dataset
from utils import load_dataset
df_train, df_test = load_dataset()
# df_train = df_train.fillna("")
Y_test = df_test.label.values

#get labeling functions (lfs)
from create_labeling_functions import *
labeling_function_list = create_labeling_functions(r'./biomimicry_functions_enumerated.csv', r'./biomimicry_function_rules.csv')

len(labeling_function_list)

if not os.path.exists('lf_analysis.pickle'):
    applier = PandasLFApplier(lfs=labeling_function_list)
    # define train and test sets
    L_train = applier.apply(df=df_train)
    L_test = applier.apply(df=df_test)

    df = LFAnalysis(L=L_train, lfs=labeling_function_list).lf_summary()
    with open('lf_analysis.pickle','wb') as f:
        pickle.dump({"lf_analysis":df, 'L_train':L_train,'L_test':L_test},f)

if os.path.exists('lf_analysis.pickle'):
    with open('lf_analysis.pickle','rb') as f:
        data = pickle.load(f)
        lf_analysis = data['lf_analysis']
        L_train = data['L_train']
        L_test = data['L_test']

majority_model = MajorityLabelVoter()
preds_train = majority_model.predict(L=L_train)

label_model = LabelModel(cardinality=665, verbose=True, device = 'gpu')
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

LFAnalysis(L=L_train, lfs=labeling_function_list).lf_summary()

df_train_filtered, preds_train_filtered = filter_unlabeled_dataframe(
    X=df_train, y=preds_train, L=L_train)

df_train["label"] = label_model.predict(L=L_train, tie_break_policy="abstain")

label_model.save("snorkel_model.pkl")

df_train.to_csv("results.csv")
