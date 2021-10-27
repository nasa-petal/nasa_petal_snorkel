# Labeling Training Data for NASA PeTaL (Periodic Table of Life)

[NASA PeTaL Project GitHub](https://github.com/nasa-petal)

# Links
 * [Overview](#overview)
 * [Files](#files)
 * [Getting Started](#getting-started)
 * [Running Snorkel](#running-snorkel)
 * [More Informaion](#more-information)
 * [Future Work](#future-work)
 * [Contact](#contact)
  
## Overview

This repository contains scripts, notebooks, data, and docs used for utilizing the snorkel machine learning model. This model uses weak supervision to label large amounts of training data using programmatic labeling functions (lfs) based on keyword 'rules'.

This README was last updated on 29 October 2021.

# Files

```snorkel.environment.yml``` environment for running snorkel with required dependencies

```petal_snorkel.py``` Main file for running snorkel

```petal_snorkel.ipynb``` Jupyter notebook for running snorkel

```biomimicry_function_rules.csv``` contains rules for 40 of the 100 biomimicry functions.

```biomimicry_functions_enumerated.csv``` contains all 100 of the biomimicry functions labeled 0-99.

```create_labeling_functions.py``` file to create keyword labeling functions (lfs)

```utils.py``` data cleaning and train/test/split of data

```snorkel_spam_test``` folder containing all the files needed to run a short test of snorkel using a spam YouTube dataset.

# Getting Started
## Environment and setup

Snorkel requires Python 3.6 or later. The entire conda environment for running snorkel can be found in  ```snorkel.environment.yml```

# Running Snorkel
Note that each script has detailed instructions in its opening comment.

## snorkel_spam_test
Get a sense of how snorkel works and run a quick data labeling tutorial using a YouTube spam comments dataset. More info can be found here: https://www.snorkel.org/use-cases/01-spam-tutorial

## labeled_data.csv
Dataset of labeled biomimicry data

## biomimicry_functions_enumerated.csv
Contains all 100 of the biomimicry functions labeled 0-99. These numbers are what snorkel recognizes in place of a biomimicry function, e.g. 'attach_permanently' = 0.

## biomimicry_function_rules.csv
Contains rules for 40 of the 100 biomimicry functions. For example, the function 'attach permanently', contains the keyword rules 'attach firmly', 'biological adhesive', and 'biological glue'.

## utils.py
Takes in data from ```labeled_data.csv``` and applies a -1 'abstain' label to each row as a default, and performs a train/test/split of the data.

## create_labeling_functions.py
Create keyword labeling functions (lfs) for every rule in ```biomimicry_function_rules.csv```

## petal_snorkel.py

# More Information
 * [Snorkel Flow](https://snorkel.ai/)
 * [Snorkel Resources](https://www.snorkel.org/resources/)

## Notable papers
 * [Snorkel: Rapid Training Data Creation with Weak Supervision](https://arxiv.org/abs/1711.10160)
 * [Data Programming: Creating Large Training Sets, Quickly](https://arxiv.org/abs/1605.07723)
 * [Learning the Structure of Generative Models without Labeled Data](https://arxiv.org/abs/1703.00854)

# Future Work
 * Integrating with Petal project
 * Etc

# Contact
For questions contact Alexandra Ralevski (alexandra.ralevski@gmail.com)





