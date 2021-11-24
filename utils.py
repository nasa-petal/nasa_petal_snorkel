from pickle import TRUE
import pandas as pd
import numpy as np
from snorkel.labeling.model import LabelModel
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
    # df = df.drop(['title', 'abstract'], axis=1)
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



def smaller_models(L_match:np.ndarray,nLabelsPerGroup:int, nOverlap:int, labels_list:Dict[int,str],df:pd.DataFrame) -> List[pd.DataFrame]:
    """Code to construct smaller snorkel models from a larger one. This will help divide the data from models that, for example, we take a large dataset containing 16 labels and break it down into a smaller dataset that that predicts 5 labels; 3 labels are unique to the model and 1 label overlaps with another model and the other remaining label (-1) is the "I don't know what label this paper is" I call this the consensus data model approach

    Args:
        L_match (np.ndarray): This comes from  applier = PandasLFApplier(lfs=labeling_function_list); L_match = applier.apply(df=df_train). Dimensions = (#PAPERS, FUNCTIONS RULES)
        nLabelsPerGroup (int): number of labels per group 
        nOverlap (int): number of overlaping labels. Defaults to > 5
        labels_list (Dict[int,str]): mapping of label to string
        df (pd.DataFrame): Pandas dataframe with all the data 

    Returns:
        (tuple): containing
            
            Labels_overlap (List[List[int]]): List of labels with overlap
            L_matches (List[np.ndarray]): List of sub L_matches specific to each Labels_overlap list 
            translations (List[Dict[int,int]]): For each model this can be used to convert the new labels to the old labels id's 
            translators_to_str (List[Dict[int,str]]): For each model this can be used to convert the key representing the label to a string with the prediction. 
            L_match (np.ndarray): This is the new Lmatch with unique values occuring in ascending order, no skips
            global_translator (Dict[int,int]): Used to convert old labels to new labels for L_match
            dfs (List[pd.DataFrame]): List of dataframes
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
    unique_labels_used = list()
    for i in range(len(labels)):
        labels_overlap.append(deepcopy(labels[i]))
        overlap_counter = 0
        for u in unique_labels:
            if u not in labels[i] and u not in unique_labels_used:
                labels_overlap[-1].append(u)
                unique_labels_used.append(u)
                overlap_counter+=1
            if overlap_counter==nOverlap:
                break
    unused_unique_labels = [u for u in unique_labels if u not in unique_labels_used]
    labels_overlap.append(unused_unique_labels)

    # Restructure L_match
    L_matches = list() # List of all the matches 
    dfs = list()
    translators = list()
    translators_to_str = list()
    for lo in labels_overlap:
        translators.append(dict(zip(lo, range(len(lo)))))
        translators_to_str.append(dict(zip(range(len(lo)), itemgetter(*lo)(labels_list))))
        L_match_mini = list()

        '''
            Note: Below we search L_match matrix for any row that matches lo. lo can be labels [0,1,6,7,8] 
            If any row matches these labels, it is automatically added to L_match mini

            The key is to end up with a matrix that contains -1,0,1,6,7,8 and a model that can predict this
        '''
        df_index = list()
        for i in range(L_match.shape[0]):               # Find all instances inside L_match that contain matches for labels in lo
            unique = np.unique(L_match[i,:]).tolist()   # Look at all the rules where this one matches 
            if any(item in lo for item in unique):      # any item in lo is in unique
                L_match_mini.append([lm if lm in lo else -1 for lm in L_match[i,:]])
                df_index.append(i)
        L_matches.append(np.array([np.array(xi) for xi in L_match_mini]))
        dfs.append(df.iloc[df_index])

    # Add in -1 to all the labels
    for i in range(len(labels_overlap)):
        labels_overlap[i].insert(0,-1)
    for i in range(len(labels)):
        labels[i].insert(0,-1)
    for i in range(len(translators)):
        translators[i] = {**{-1:-1}, **translators[i]}
    for i in range(len(translators_to_str)):
        translators_to_str[i] = {**{-1:'no_match'},**translators_to_str[i]}

    # Normalize the data convert numbers that skip to incremental 
    for i in range(len(labels_overlap)):
        for j in range(len(labels_overlap[i])):
            key = labels_overlap[i][j]
            value = translators[i][key] # Swap with normalized label 
            labels_overlap[i][j] = value
            L_matches[i] = np.where(L_matches[i] == key, value, L_matches[i])
    
    # Create global comparison
    global_translator = dict(zip(unique_labels,range(len(unique_labels))))
    global_translator[-1] = -1
    for key,value in global_translator.items():
        L_match = np.where(L_match == key, value, L_match)
        
    return labels_overlap, L_matches, translators, translators_to_str, L_match, global_translator, dfs


def evaluate_model(L:np.ndarray,model:LabelModel,translator:Dict[int,str], model_index:int) -> Dict[str,float]:
    """[summary]

    Args:
        L (np.ndarray): [description]
        model (LabelModel): [description]
        translator (Dict[int,str]): [description]
        model_index (int): index of model use as a reference 

    Returns:
        Dict[str,float]: [description]
    """
    n_papers,_ = L.shape
    probs_train = model.predict_proba(L=L)
    results = list()
    keys = list(translator.keys())
    # Loop through all papers and take top 3 
    for i in range(n_papers):
        j = 0
        results.append(list())
        for k,v in translator.items():
            results[i].append(
                    {
                        'label':v,'probability': probs_train[i,j],'model_index':model_index
                    }
                )
            j+=1
    return results

