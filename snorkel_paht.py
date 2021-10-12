import pandas as pd 
from snorkel.labeling import LabelingFunction, PandasLFApplier

'''
    Useful Functions 
'''
def keyword_lookup(x,bio_functions:pd.DataFrame,bio_function_rules:pd.DataFrame):
    """Returns the id cooresponding to the label

    Args:
        x (str): some phrase

    Returns:
        int: the id
    """
    for i in range(len(bio_functions)):
        label_name = bio_functions.iloc[i]['function'] 
        label_id = bio_functions.iloc[i]['function_enumerated']        
        
        label_rule_name = label_name + "_rules"
        if label_rule_name in list(bio_function_rules.columns):
            phrases_to_look_for = bio_function_rules[label_rule_name].to_list()
            for phrase in phrases_to_look_for:
                # now you could make a counter and see the percentage match so if 10/20 phrases are in the text/abstract then you return the
                if phrase in x.text.lower():     
                    return label_id
        else:
            print(f"Label {label_name} does not have rules associated with it")

'''
    Main Code 
'''
def create_labeling_functions(bio_file:str, bio_rules:str):
    bio_functions = pd.read_csv(bio_file)
    function_rule_phrases = pd.read_csv(bio_rules)
    
    labeling_function_list = list()

    # Create a Labeling Function for each Label
    for i in range(len(bio_functions)):
        label_name = bio_functions.iloc[i]['function']
        labeling_function = LabelingFunction(name=f"keyword_{label_name}", f=keyword_lookup,
                resources={"bio_functions":bio_functions,"bio_function_rules":function_rule_phrases})
        labeling_function_list.append(labeling_function)
    
    return labeling_function_list