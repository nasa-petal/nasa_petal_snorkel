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
from utils import smaller_models
from create_labeling_functions import create_labeling_functions
import pickle, os
import pandas as pd 

#import csv file and load train/test/split of dataset
from utils import load_dataset

df = load_dataset()
#get labeling functions (lfs)

labeling_function_list = create_labeling_functions(r'./biomimicry_functions_enumerated.csv', r'./biomimicry_function_rules.csv')
df_bio = pd.read_csv(r'./biomimicry_functions_enumerated.csv')
labels = dict(zip(df_bio['function_enumerated'].tolist(),df_bio['function'].tolist()))

applier = PandasLFApplier(lfs=labeling_function_list)
L_match = applier.apply(df=df)
smaller_models(L_match,5,2,labels_list=labels)



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
