from pickle import TRUE
import pandas as pd
import numpy as np
from pandas.core.algorithms import unique
from sklearn.model_selection import train_test_split
from typing import Dict, List
from itertools import combinations
from copy import deepcopy
from operator import itemgetter

def load_dataset(load_train_labels: bool = False, split_dev_valid: bool = False):
    filename = r"labeled_data.csv"
    df = pd.read_csv(filename)
    #lowercase column names
    df.columns = map(str.lower, df.columns)
    #comnine title and abstract columns into one and drop columns
    df['text'] = df['title'] + ' ' + df['abstract']
    df = df.drop(['title', 'abstract'], axis=1)
    return df 


    # df_train = df.sample(frac=0.6, random_state=123).reset_index(drop=True)
    # df_dev = df_train.sample(frac=0.2, random_state=123)

    # if not load_train_labels:
    #     df_train['label'] = np.ones(len(df_train['label'])) * -1
    
    # df_valid_test = df_train.sample(frac=0.5, random_state=123)
    # df_valid, df_test = train_test_split(df_valid_test, random_state=123, stratify=df_valid_test.label)

    # if split_dev_valid:
    #     return df_train, df_dev, df_valid, df_test
    # else:
    #     return df_train, df_test
    
    """
        Returns
            (List)
    """



def smaller_models(L_match:np.ndarray,nLabelsPerGroup:int, nOverlap:int, labels_list:Dict[int,str]) -> List[pd.DataFrame]:
    """Code to construct smaller snorkel models from a larger one. This will help divide the data from models that, for example, pick 16 labels to something that predicts 5 labels 3 are unique to the model and 1 is overlap with another model and the other label is NOT LABELS 1-4

    Args:
        L_match (np.ndarray): This comes from  applier = PandasLFApplier(lfs=labeling_function_list); L_match = applier.apply(df=df_train). Dimensions = (#PAPERS, FUNCTIONS RULES)
        nLabelsPerGroup (int): number of labels per group 
        nOverlap (int): number of overlaping labels. Defaults to > 5
        labels_list (Dict[int,str]): mapping of label to string

    Returns:
        (tuple): containing
            
            Labels_overlap (List[List[int]]): List of labels with overlap
            L_matches (List[np.ndarray]): List of sub L_matches specific to each Labels_overlap list 
            translations (List[Dict[int,int]]): translations to normalize both 
    """
    assert nLabelsPerGroup+1-nOverlap>2, "Need to have 2+ unique labels per group."

    unique_labels = np.unique(L_match).tolist()  # Here there maybe some skips 1,2,5,9,12,13 so 6 total unique labels
    unique_labels.remove(-1)
    unique_labels_backup = deepcopy(unique_labels)
    # if you want 1 overlap and 3 labels per group then we want [1,2,5] [5,9,12] [12,13,1]

    # Create the list of labels without overlap 
    labels = list()
    while True:
        label = list()
        for u in unique_labels:
            if u not in labels and len(label) < nLabelsPerGroup-nOverlap:
                skip = False
                for lu in labels:
                    if u in lu:
                        skip = True    
                if not skip:
                    label.append(u)
            elif len(labels) >= nLabelsPerGroup-nOverlap:
                break
        if not label:
            break
        labels.append(label)
    # Lets add the overlap
    labels_overlap = list() 
    for n in range(nOverlap):
        if n == 0:
            for i in range(len(labels)):
                for u in unique_labels:
                    if u not in labels[i]:
                        labels_overlap.append(deepcopy(labels[i]))
                        labels_overlap[-1].append(u)
        else:
            for lo in range(len(labels_overlap)):
                for u in unique_labels:
                    if u not in labels_overlap[lo]:
                        labels_overlap[lo].append(u)
                        unique_labels.remove(u)
                        break
        unique_labels = deepcopy(unique_labels_backup)   

    # Restructure L_match
    L_matches = list() # List of all the matches 
    translators = list()
    translators_to_str = list()
    for lo in labels_overlap:
        translators.append(dict(zip(lo, range(len(lo)))))
        translators_to_str.append(dict(zip(range(len(lo)), itemgetter(*lo)(labels_list))))
        L_match_mini = list()
        for i in range(L_match.shape[0]): # Find all instances inside L_match that contain matches for labels in lo
            unique = np.unique(L_match[i,:]).tolist() # Look at all the rules where this one matches 
            if any(item in lo for item in unique): # any item in lo is in unique
                L_match_mini.append([lm if lm in lo else -1 for lm in L_match[i,:]])
        L_matches.append(np.array([np.array(xi) for xi in L_match_mini]))
    
    # Add in -1 to all the labels
    for i in range(len(labels_overlap)):
        labels_overlap[i].insert(0,-1)
    for i in range(len(labels)):
        labels[i].insert(0,-1)
    for i in range(len(translators)):
        translators[i][-1] = -1
    for i in range(len(translators_to_str)):
        translators_to_str[i][-1] = 'no_match'

    # Normalize the data convert numbers that skip to incremental 
    for i in range(len(labels_overlap)):
        for j in range(len(labels_overlap[i])):
            key = labels_overlap[i][j]
            value = translators[i][key] # Swap with normalized label 
            labels_overlap[i][j] = value
            L_matches[i] = np.where(L_matches[i] == key, value, L_matches[i])
    
    # Create global comparison
    global_translator = dict(zip(unique_labels,range(len(unique_labels))))
    for key,value in global_translator.items():
        L_match = np.where(L_match == key, value, L_match)
        
    return labels_overlap, L_matches, translators, translators_to_str, L_match, global_translator