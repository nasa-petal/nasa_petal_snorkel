import pandas as pd 
from snorkel.labeling import LabelingFunction
import itertools
import math
from snorkel.labeling.lf.core import labeling_function
import numpy as np 
'''
    Useful Functions 
'''
def keyword_lookup(x,phrase_to_match:str, label_id:int):
    """Returns the id corresponding to the label

    Args:
        phrase_to_match (str): some phrase that we need to match
        label_id (int): id of label to use for this match

    Returns:
        (int): label id if match or -1 if no match 
    """
    if phrase_to_match.lower() in x.text.lower():     
        return label_id
    else:
        return -1
        

'''
    Main Code 
'''


def create_labeling_functions(bio_file:pd.DataFrame, bio_rules:pd.DataFrame):
    """create a list of labeling functions

    Args:
        bio_file (pd.DataFrame): a list of all the biomimicry functions
        bio_rules (pd.DataFrame): a list of all the 'rules' for each biomimicry function

    Returns:
        labeling_function_list: a list of all the labeling function 'rules' corresponding to each biomimicry function
    """
    bio_file = pd.read_csv(bio_file)
    bio_rules = pd.read_csv(bio_rules)

    names_used = list()
    labeling_function_list = list()
    
    #get a list of all the rules
    for i in range(len(bio_file)):

        label_name = bio_file.iloc[i]['function']
        label_id = bio_file.iloc[i]['function_enumerated']
        label_rule_name = label_name + "_rules"

        if label_rule_name in list(bio_rules.columns):
            underscore_list = []
            phrases_lst = bio_rules[label_rule_name].to_list()
            
            #remove blank cells and keep unique values 
            rules_no_na = list(set([x for x in phrases_lst if not pd.isnull(x)]))
            
            #add underscore to rules
            for item in rules_no_na:
                item = item.replace(" ", "_")
                underscore_list.append(item)
            #create labeling function for each rule
            for phrase in underscore_list:
                function_name = f"keyword_{label_id}_{phrase}"
                if (function_name not in names_used):
                    labeling_function = LabelingFunction(name=function_name, f=keyword_lookup,
                                    resources={"phrase_to_match":phrase, "label_id":label_id})
                    labeling_function_list.append(labeling_function)
                    names_used.append(function_name)
    
    return labeling_function_list
    




        
