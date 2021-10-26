# PeTaL Snorkel 
## Overview

This repository contains scripts, notebooks, data, and docs used for utilizing the snorkel machine learning model. This model uses weak supervision to label large amounts of training data using programmatic labeling functions (lfs) based on keyword 'rules'.

This README was last updated on 29 October 2021.

# Files

```petal_snorkel.py``` Main file for running snorkel

```petal_snorkel.ipynb``` Jupyter notebook for running snorkel

```biomimicry_function_rules.csv``` contains rules for 40 of the 100 biomimicry functions. For example, the function 'attach permanently', contains the keyword rules 'attach firmly', 'biological adhesive', and 'biological glue'.

```biomimicry_functions_enumerated.csv``` contains all 100 of the biomimicry functions labeled 0-99.

```create_labeling_functions.py``` file to create keyword labeling functions (lfs)

```utils.py``` data cleaning and train/test/split of data

```snorkel_spam_test``` folder containing all the files needed to run a short test of snorkel using a spam YouTube dataset. More details here: https://www.snorkel.org/use-cases/01-spam-tutorial


