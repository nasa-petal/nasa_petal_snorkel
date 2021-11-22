'''
    This example shows how to use the consensus label model approach to predict the "label" of any given text using snorkel
'''
import sys
sys.path.insert(0,'../snorkel')
from utils import smaller_models
import wget
import os.path as osp 
import pickle, json
import pandas as pd 
from tqdm import trange
import numpy as np 
from ast import literal_eval
golden_json_url = 'https://raw.githubusercontent.com/nasa-petal/data-collection-and-prep/main/golden.json'

filename = 'golden.json'
large_model = 'large_model_trained.pickle'
small_models = 'small_models_trained.pickle'
if not osp.exists(filename):
    wget.download(golden_json_url)

with open(filename,'r') as f:
    golden_json = json.load(f)

if osp.exists(small_models):
    with open(small_models,'rb') as f:
        smaller_model_data = pickle.load(f)
'''
    Evaluate Golden.json with smaller models 
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
    from create_labeling_functions import create_labeling_functions
    labeling_function_list = create_labeling_functions(r'./biomimicry_functions_enumerated.csv', r'./biomimicry_function_rules.csv')
    from snorkel.labeling import PandasLFApplier

    applier = PandasLFApplier(lfs=labeling_function_list)
    L_Golden = applier.apply(df=df)

    with open('golden_lf.pickle','wb') as f:
        pickle.dump({'L_Golden':L_Golden},f)

with open('golden_lf.pickle','rb') as f:
    data = pickle.load(f) 
    L_Golden = data['L_Golden']
    print("Unique Matches in golden.json: ")
    print(*np.unique(L_Golden).tolist(), sep = ", ")

'''
    Evaluation using smaller models
'''
threshold = 0.5
model_predictions = list()
for i in trange(len(smaller_model_data['Label_models']), desc="looping through smaller models"):
    model = smaller_model_data['Label_models'][i]
    translator = smaller_model_data['translators'][i]
    translators_to_str = smaller_model_data['translators_to_str'][i]
    probs_train = model.predict_proba(L=L_Golden)
    if np.any(probs_train>0.5):
        result = np.where(probs_train > 0.5)
        for r,c in zip(result[0], result[1]):
            paper_id=golden_json[r]['paper']
            doi=golden_json[r]['doi']
            label_local_id = model
            model_predictions.append({})
        print('check')
        # model_predictions.append({paper})


    
'''
    Evaluate using larger model
'''
