from snorkel.labeling import labeling_function
# from snorkel.labeling import LabelingFunction
import pandas as pd
import re

# def keyword_lookup(x, keywords, label):
#     if any(word in x.text.lower() for word in keywords):
#         return label
#     return abstain
    
# Load Enumerations
# def clean_text(text):
#     new_text = re.sub(" and/or | through/on | through | or |/|\s", "_", text).lower()
#     return new_text

enum_bio = pd.read_csv(r"C:\Users\ARalevski\Documents\Petal\Snorkel-PeTaL\biomimicry_functions_enumerated.csv")
# enum_bio["function"] = enum_bio["function"].apply(clean_text)
enum_bio = enum_bio.set_index("function")
abstain = -1

# Load phrase heuristics
function_rule_phrases = pd.read_csv(r"C:\Users\ARalevski\Documents\Petal\Snorkel-PeTaL\biomimicry_function_rules.csv")
function_rule_phrases = function_rule_phrases.fillna("")

#Protect from living/non-living threats
##Protect from living threats
###Protect from animals
@labeling_function()
def lf_protect_from_animals(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['protect_from_animals_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["protect_from_animals", "function_enumerated"]
    return abstain

###Protect from plants
@labeling_function()
def lf_protect_from_plants(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['protect_from_plants_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["protect_from_plants", "function_enumerated"]
    return abstain
###Protect from fungi
@labeling_function()
def lf_protect_from_fungi(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['protect_from_fungi_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["protect_from_fungi", "function_enumerated"]
    return abstain
###Protect from microbes
@labeling_function()
def lf_protect_from_microbes(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['protect_from_microbes_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["protect_from_microbes", "function_enumerated"]
    return abstain

##Protect from non-living threats
###Protect from solids
@labeling_function()
def lf_protect_from_solids(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['protect_from_solids_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["protect_from_solids", "function_enumerated"]
    return abstain


###Protect from excess liquids
@labeling_function()
def lf_protect_from_excess_liquids(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['protect_from_excess_liquids_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["protect_from_excess_liquids", "function_enumerated"]
    return abstain

# def make_keyword_lf(keywords, label='protect_from_excess_liquids'):
#     keywords = function_rule_phrases['protect_from_excess_liquids_rules'].to_list()
#     return LabelingFunction(
#         name=f"keyword_{keywords[0]}",
#         f=keyword_lookup,
#         resources=dict(keywords=keywords, label=label),
#     )


###Protect from wind
@labeling_function()
def lf_protect_from_wind(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['protect_from_wind_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["protect_from_wind", "function_enumerated"]
    return abstain
###Protect from temperature
@labeling_function()
def lf_protect_from_temperature(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['protect_from_temperature_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["protect_from_temperature", "function_enumerated"]
    return abstain
###Protect from fire
@labeling_function()
def lf_protect_from_fire(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['protect_from_fire_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["protect_from_fire", "function_enumerated"]
    return abstain
###Protect from ice
@labeling_function()
def lf_protect_from_ice(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['protect_from_ice_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["protect_from_ice", "function_enumerated"]
    return abstain
###Protect from light
@labeling_function()
def lf_protect_from_light(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['protect_from_light_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["protect_from_light", "function_enumerated"]
    return abstain
###Protect from chemicals
@labeling_function()
def lf_protect_from_chemicals(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['protect_from_chemicals_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["protect_from_chemicals", "function_enumerated"]
    return abstain
###Protect from radiation
@labeling_function()
def lf_protect_from_radiation(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['protect_from_radiation_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["protect_from_radiation", "function_enumerated"]
    return abstain
