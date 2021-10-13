from snorkel.labeling import labeling_function
import pandas as pd
import re

# Load Enumerations
# def clean_text(text):
#     new_text = re.sub(" and/or | through/on | through | or |/|\s", "_", text).lower()
#     return new_text

enum_bio = pd.read_csv("formatted_enums.csv")
# enum_bio["function"] = enum_bio["function"].apply(clean_text)
enum_bio = enum_bio.set_index("function")
abstain = -1

# Load phrase heuristics
function_rule_phrases = pd.read_csv("biomimicry_function_rules.csv")
function_rule_phrases = function_rule_phrases.fillna("")
#Attach
##Attach permanently
@labeling_function()
def lf_attach_permanently(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['attach_permanently_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["attach_permanently", "function_enumerated"]
    return abstain

##Attach temporarily
@labeling_function()
def lf_attach_temporarily(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['attach_temporarily_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["attach_temporarily", "function_enumerated"]
    return abstain

    
#Move on/through solids, liquids, gases
##Passive movement
###Passively move through/on solids
@labeling_function()
def lf_passively_move_solids(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['passively_move_solids_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["passively_move_solids", "function_enumerated"]
    return abstain
###Passively move through/on liquids
@labeling_function()
def lf_passively_move_liquids(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['passively_move_liquids_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["passively_move_liquids", "function_enumerated"]
    return abstain
###Passively move through gases
@labeling_function()
def lf_passively_move_gases(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['passively_move_gases_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["passively_move_gases", "function_enumerated"]
    return abstain
###Passively move through/on granular media
@labeling_function()
def lf_passively_move_granular(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['passively_move_granular_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["passively_move_granular", "function_enumerated"]
    return abstain

##Active movement
###Actively move through/on solids
@labeling_function()
def lf_actively_move_solids(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['actively_move_solids_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["actively_move_solids", "function_enumerated"]
    return abstain
###Actively move through/on liquids
@labeling_function()
def lf_actively_move_liquids(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['actively_move_liquids_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["actively_move_liquids", "function_enumerated"]
    return abstain
###Actively move through gases
@labeling_function()
def lf_actively_move_gases(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['actively_move_gases_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["actively_move_gases", "function_enumerated"]
    return abstain
###Actively move through/on granular media
@labeling_function()
def lf_actively_move_granular(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['actively_move_granular_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["actively_move_granular", "function_enumerated"]
    return abstain

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
###Protect from loss of liquids
@labeling_function()
def lf_protect_from_loss_liquids(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['protect_from_loss_liquids_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["protect_from_loss_liquids", "function_enumerated"]
    return abstain
###Protect from gases
@labeling_function()
def lf_protect_from_gases(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['protect_from_gases_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["protect_from_gases", "function_enumerated"]
    return abstain
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

#Manage mechanical forces
##Manage external forces
###Manage impact
@labeling_function()
def lf_manage_impact(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['manage_impact_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["manage_impact", "function_enumerated"]
    return abstain
###Regulate drag/turbulence
@labeling_function()
def lf_regulate_drag_turbulence(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['regulate_drag_turbulence_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["regulate_drag_turbulence", "function_enumerated"]
    return abstain

##Manage stress/strain
###Prevent buckling
@labeling_function()
def lf_prevent_buckling(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['prevent_buckling_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["prevent_buckling", "function_enumerated"]
    return abstain
###Prevent fracture/rupture
@labeling_function()
def lf_prevent_fracture_rupture(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['prevent_fracture_rupture_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["prevent_fracture_rupture", "function_enumerated"]
    return abstain
###Manage shear
@labeling_function()
def lf_manage_shear(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['manage_shear_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["manage_shear", "function_enumerated"]
    return abstain
###Prevent or allow deformation
@labeling_function()
def lf_prevent_allow_deformation(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['prevent_allow_deformation_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["prevent_allow_deformation", "function_enumerated"]
    return abstain

##Prevent degradation
###Prevent fatigue
@labeling_function()
def lf_prevent_fatigue(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['prevent_fatigue_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["prevent_fatigue", "function_enumerated"]
    return abstain
###Control erosion
@labeling_function()
def lf_control_erosion(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['control_erosion_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["control_erosion", "function_enumerated"]
    return abstain
###Regulate wear
@labeling_function()
def lf_regulate_wear(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['regulate_wear_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["regulate_wear", "function_enumerated"]
    return abstain

##Change material properties
@labeling_function()
def lf_change_material_properties(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['change_material_properties_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["change_material_properties", "function_enumerated"]
    return abstain

#Sustain ecological community
##Individual benefit
###Regulate reproduction or growth
@labeling_function()
def lf_regulate_reproduction_growth(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['regulate_reproduction_growth_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["regulate_reproduction_growth", "function_enumerated"]
    return abstain
###Self-replicate
@labeling_function()
def lf_self_replicate(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['self_replicate_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["self_replicate", "function_enumerated"]
    return abstain
###Maintain homeostasis/equillibrium
@labeling_function()
def lf_maintain_homeostasis_equillibrium(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['maintain_homeostasis_equillibrium_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["maintain_homeostasis_equillibrium", "function_enumerated"]
    return abstain

##Group benefit
###Coordinate by self-organization
@labeling_function()
def lf_coordinate_self_organization(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['coordinate_self_organization_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["coordinate_self_organization", "function_enumerated"]
    return abstain
###Cooperate within or between species
@labeling_function()
def lf_cooperate_within_between_species(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['cooperate_within_between_species_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["cooperate_within_between_species", "function_enumerated"]
    return abstain
###Compete within or between species
@labeling_function()
def lf_compete_within_between_species(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['compete_within_between_species_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["compete_within_between_species", "function_enumerated"]
    return abstain
###Manage environmental disturbance in a community
@labeling_function()
def lf_manage_environmental_disturbance(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['manage_environmental_disturbance_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["manage_environmental_disturbance", "function_enumerated"]
    return abstain
###Manage populations/pests/diseases
@labeling_function()
def lf_manage_populations_pests_diseases(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['manage_populations_pests_diseases_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["manage_populations_pests_diseases", "function_enumerated"]
    return abstain
###Maintain biodiversity
@labeling_function()
def lf_maintain_biodiversity(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['maintain_biodiversity_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["maintain_biodiversity", "function_enumerated"]
    return abstain

#Chemically assemble or break down
##Chemically assemble
###Chemically assemble inorganic compounds
@labeling_function()
def lf_chemically_assemble_inorganic_compounds(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['chemically_assemble_inorganic_compounds_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["chemically_assemble_inorganic_compounds", "function_enumerated"]
    return abstain
###Chemically assemble organic compounds
@labeling_function()
def lf_chemically_assemble_organic_compounds(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['chemically_assemble_organic_compounds_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["chemically_assemble_organic_compounds", "function_enumerated"]
    return abstain
###Self-assemble
@labeling_function()
def lf_self_assemble(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['self_assemble_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["self_assemble", "function_enumerated"]
    return abstain

##Chemically break down
###Chemically break down inorganic compounds
@labeling_function()
def lf_chemically_break_down_inorganic_compounds(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['chemically_break_down_inorganic_compounds_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["chemically_break_down_inorganic_compounds", "function_enumerated"]
    return abstain
###Chemically break down organic compounds
@labeling_function()
def lf_chemically_break_down_organic_compounds(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['chemically_break_down_organic_compounds_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["chemically_break_down_organic_compounds", "function_enumerated"]
    return abstain

#Modify or convert energy
##Modify/convert electrical energy
@labeling_function()
def lf_modify_convert_electrical_energy(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['modify_convert_electrical_energy_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["modify_convert_electrical_energy", "function_enumerated"]
    return abstain
##Modify/convert magnetic energy
@labeling_function()
def lf_modify_convert_magnetic_energy(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['modify_convert_magnetic_energy_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["modify_convert_magnetic_energy", "function_enumerated"]
    return abstain
##Modify/convert chemical energy
@labeling_function()
def lf_modify_convert_chemical_energy(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['modify_convert_chemical_energy_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["modify_convert_chemical_energy", "function_enumerated"]
    return abstain
##Modify/convert mechanical energy
@labeling_function()
def lf_modify_convert_mechanical_energy(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['modify_convert_mechanical_energy_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["modify_convert_mechanical_energy", "function_enumerated"]
    return abstain
##Modify/convert thermal energy
@labeling_function()
def lf_modify_convert_thermal_energy(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['modify_convert_thermal_energy_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["modify_convert_thermal_energy", "function_enumerated"]
    return abstain
##Modify/convert light energy
@labeling_function()
def lf_modify_convert_light_energy(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['modify_convert_light_energy_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["modify_convert_light_energy", "function_enumerated"]
    return abstain

#Physically assemble/disassemble
##Physically assemble structure
@labeling_function()
def lf_physically_assemble_structure(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['physically_assemble_structure_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["physically_assemble_structure", "function_enumerated"]
    return abstain
##Break down structure
@labeling_function()
def lf_break_down_structure(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['break_down_structure_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["break_down_structure", "function_enumerated"]
    return abstain
##Optimize shape/materials
@labeling_function()
def lf_optimize_shape_materials(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['optimize_shape_materials_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["optimize_shape_materials", "function_enumerated"]
    return abstain

#Sense, send or process information
##Send signals
###Send light signals in the visible spectrum
@labeling_function()
def lf_send_light_visible_spectrum(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['send_light_visible_spectrum_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["send_light_visible_spectrum", "function_enumerated"]
    return abstain
###Send light signals in the non-visible spectrum
@labeling_function()
def lf_send_light_nonvisible_spectrum(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['send_light_nonvisible_spectrum_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["send_light_nonvisible_spectrum", "function_enumerated"]
    return abstain
###Send sound signals
@labeling_function()
def lf_send_sound_signals(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['send_sound_signals_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["send_sound_signals", "function_enumerated"]
    return abstain
###Send tactile signals
@labeling_function()
def lf_send_tactile_signals(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['send_tactile_signals_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["send_tactile_signals", "function_enumerated"]
    return abstain
###Send chemical signals 
@labeling_function()
def lf_send_chemical_signals(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['send_chemical_signals_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["send_chemical_signals", "function_enumerated"]
    return abstain
###Send vibratory signals
@labeling_function()
def lf_send_vibratory_signals(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['send_vibratory_signals_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["send_vibratory_signals", "function_enumerated"]
    return abstain
###Send electrical/magnetic signals
@labeling_function()
def lf_send_electrical_magnetic_signals(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['send_electrical_magnetic_signals_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["send_electrical_magnetic_signals", "function_enumerated"]
    return abstain

##Process signals
###Differentiate signal from noise
@labeling_function()
def lf_differentiate_signal_noise(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['differentiate_signal_noise_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["differentiate_signal_noise", "function_enumerated"]
    return abstain
###Convert signals
@labeling_function()
def lf_convert_signals(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['convert_signals_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["convert_signals", "function_enumerated"]
    return abstain
###Respond to signals
@labeling_function()
def lf_respond_signals(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['respond_signals_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["respond_signals", "function_enumerated"]
    return abstain

##Sense signals/environmental cues
###Sense light in the visible spectrum
@labeling_function()
def lf_sense_light_visible_spectrum(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['sense_light_visible_spectrum_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["sense_light_visible_spectrum", "function_enumerated"]
    return abstain
###Sense light in the non-visible spectrum
@labeling_function()
def lf_sense_light_nonvisible_spectrum(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['sense_light_nonvisible_spectrum_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["sense_light_nonvisible_spectrum", "function_enumerated"]
    return abstain
###Sense electricity/magnetism
@labeling_function()
def lf_sense_electricity_magnetism(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['sense_electricity_magnetism_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["sense_electricity_magnetism", "function_enumerated"]
    return abstain
###Sense disease in a living system
@labeling_function()
def lf_sense_disease(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['sense_disease_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["sense_disease", "function_enumerated"]
    return abstain
###Sense touch and mechanical forces
@labeling_function()
def lf_sense_touch_mechanical_forces(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['sense_touch_mechanical_forces_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["sense_touch_mechanical_forces", "function_enumerated"]
    return abstain
###Sense chemicals
@labeling_function()
def lf_sense_chemicals(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['sense_chemicals_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["sense_chemicals", "function_enumerated"]
    return abstain 
###Sense atmospheric conditions
@labeling_function()
def lf_sense_atmospheric_conditions(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['sense_atmospheric_conditions_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["sense_atmospheric_conditions", "function_enumerated"]
    return abstain
###Sense sound/vibrations
@labeling_function()
def lf_sense_sound_vibrations(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['sense_sound_vibrations_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["sense_sound_vibrations", "function_enumerated"]
    return abstain
###Sense temperature cues
@labeling_function()
def lf_sense_temperature_cues(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['sense_temperature_cues_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["sense_temperature_cues", "function_enumerated"]
    return abstain
###Sense motion
@labeling_function()
def lf_sense_motion(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['sense_motion_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["sense_motion", "function_enumerated"]
    return abstain
###Sense spatial awareness/balance/orientation
@labeling_function()
def lf_sense_spatial_awareness(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['sense_spatial_awareness_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["sense_spatial_awareness", "function_enumerated"]
    return abstain
###Sense shape and/or pattern
@labeling_function()
def lf_sense_shape_pattern(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['sense_shape_pattern_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["sense_shape_pattern", "function_enumerated"]
    return abstain

#Manipulate solids, liquids, gases or energy
##Capture resources
###Capture solids
@labeling_function()
def lf_capture_solids(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['capture_solids_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["capture_solids", "function_enumerated"]
    return abstain
###Capture liquids
@labeling_function()
def lf_capture_liquids(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['capture_liquids_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["capture_liquids", "function_enumerated"]
    return abstain
###Capture gases
@labeling_function()
def lf_capture_gases(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['capture_gases_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["capture_gases", "function_enumerated"]
    return abstain
###Capture energy
@labeling_function()
def lf_capture_energy(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['capture_energy_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["capture_energy", "function_enumerated"]
    return abstain

##Absorb and/or filter resources
###Absorb and/or filter solids
@labeling_function()
def lf_absorb_filter_solids(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['absorb_filter_solids_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["absorb_filter_solids", "function_enumerated"]
    return abstain
###Absorb and/or filter liquids
@labeling_function()
def lf_absorb_filter_liquids(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['absorb_filter_liquids_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["absorb_filter_liquids", "function_enumerated"]
    return abstain

###Absorb and/or filter gases
@labeling_function()
def lf_absorb_filter_gases(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['absorb_filter_gases_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["absorb_filter_gases", "function_enumerated"]
    return abstain

##Store resources
###Store solids
@labeling_function()
def lf_store_solids(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['store_solids_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["store_solids", "function_enumerated"]
    return abstain
###Store liquids
@labeling_function()
def lf_store_liquids(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['store_liquids_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["store_liquids", "function_enumerated"]
    return abstain
###Store gases
@labeling_function()
def lf_store_gases(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['store_gases_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["store_gases", "function_enumerated"]
    return abstain
###Store energy
@labeling_function()
def lf_store_energy(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['store_energy_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["store_energy", "function_enumerated"]
    return abstain

##Distribute or expel resources
###Distribute or expel solids
@labeling_function()
def lf_distribute_expel_solids(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['distribute_expel_solids_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["distribute_expel_solids", "function_enumerated"]
    return abstain
###Distribute or expel liquids
@labeling_function()
def lf_distribute_expel_liquids(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['distribute_expel_liquids_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["distribute_expel_liquids", "function_enumerated"]
    return abstain
###Distribute or expel gases
@labeling_function()
def lf_distribute_expel_gases(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['distribute_expel_gases_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["distribute_expel_gases", "function_enumerated"]
    return abstain
###Distribute or expel energy
@labeling_function()
def lf_distribute_expel_energy(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['distribute_expel_energy_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["distribute_expel_energy", "function_enumerated"]
    return abstain

##Detox/purify
@labeling_function()
def lf_detox_purify(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['detox_purify_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["detox_purify", "function_enumerated"]
    return abstain

#Change size or color
##Modify color
###Change strucutral color
@labeling_function()
def lf_change_structural_color(doc):
    """Check if any of these phrases are in the document"""
    colors = ['red', 'green', 'blue', 'indigo', 'violet', 'white', 'black']
    phrases = function_rule_phrases['change_structural_color_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["change_structural_color", "function_enumerated"]
    return abstain
###Change chemical color/pigmentation
@labeling_function()
def lf_change_chemical_color_pigmentation(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['change_chemical_color_pigmentation_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["change_chemical_color_pigmentation", "function_enumerated"]
    return abstain
###Camouflage/mimicry
@labeling_function()
def lf_camouflage_mimicry(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['camouflage_mimicry_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["camouflage_mimicry", "function_enumerated"]
    return abstain
###Change size/shape
@labeling_function()
def lf_change_size_shape(doc):
    """Check if any of these phrases are in the document"""
    phrases = function_rule_phrases['change_size_shape_rules'].to_list()
    for phrase in phrases:
        if phrase in doc.title or phrase in doc.abstract:
            return enum_bio.loc["change_size_shape", "function_enumerated"]
    return abstain