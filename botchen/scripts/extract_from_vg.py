'''
This script transforms Visual Genome (VG) ideallanguage and region graph data into a structured conversational format designed for training 
chatbot systems like Botchen. 

It processes VG data in 5 final formats:
In five final formats:
1. Logic → Logic 
2. Logic → Surface
3. Surface → Logic
4. Surface → Surface
5. Sandwich (mixed format)

Main functionalities:
- Extraction: Parses data from Visual Genome into logic and surface utterances.
- Mapping: Links logical representations to surface forms using entity alignment.
- Corpus Augmentation: Automatically finds similar situations based on content and entity overlap, enabling dataset expansion.
- Permutations Applies semantic substitutions using ConceptNet synonyms and hypernyms.
- Prompting: Converts surface and logic utterances into prompt formats.
- Train/Test Split: Supports balanced splitting of original + augmented data.

With files all written down it is:
data/
├── training/
│ ├── original/
│ ├── augmented/
│ ├── permuted_files/
│ └── prompt_files/
└── testing/
├── augmented/
├── permuted_files/
└── prompt_files/

Each subfolder may contain:
- `*_logic_to_logic.txt`
- `*_logic_to_surface.txt`
- `*_surface_to_logic.txt`
- `*_surface_to_surface.txt`
- `*_sandwich.txt`
And their permuted or prompt-based variants.

Parameters "BEGINNING DYNAMICAL PARAMETERS WHICH THE USER CAN CHANGE"

Key options/parameters:
    - `ids`: List of `(vg_id, new_id)` pairs to extract. `E.g. ids = [(1, 1), (3, 2), (4, 3), (71, 4), (9, 5)]`
    - `substitution_terms_list`: List of terms to substitute with synonyms or hypernyms from ConceptNet.
    - `extend_corpus`: If `True`, adds similar situations to increase data. Similarity is measured in terms of common entities.
    - `permutation_flag`: Applies word substitutions from [https://conceptnet.io/][ConceptNet]. Random substitution from a list of hypernyms and synonims. You can select which terms to substitute (`e.g. substitution_terms_list = ['car','jacket','shirt', 'man','woman', 'tree','road', 'bicycle']`).
    - `training_and_test_sets`. Boolean. Whether to split data into train/test sets
    - `limited`, `limited_max_utterances`: Limit utterances per situation (for small data testing).
    - `test_mode`, `test_max_situations`: Extract a subset of situations for testing.
    - `write_all_files`: If `True`, saves all generated file versions

Option parameter for corpus increasing:
    - min_referent_overlap_ratio Minimum proportion of referent entities that must appear in a target situation (i.e. we apply this to referent situations, e.g. *1 if the referent situation is as such)
    - min_target_overlap_ratio Minimum proportion of target entities that must match referent entities (i.e. we apply this to all the *10 situations which we are finding similar to a referent situation *1)
    - min_content_length # Minimum number of characters in a situation's content
    - max_content_length Maximum number of characters in a situation's content
    - max_per_referent Maximum number of similar situations to extract per referent situation (e.g. we take *10* situations similar to situation 1, *10* to situation 2)
    - train_split_ratio Percentage of training and testing sets

Dependencies 
    - Python3+
    - `./scripts/utils_extract_from_corpora.py`
    - `./scripts/utils_permutation_prompt.py`
    - External files:
      - `./data/ideallanguage.txt` (from unzipping `./data/ideallanguage.zip`)
      - `../dsc/region_graphs.json.dsc`(from running the python files in the bigger folder).
'''

import logging
logging.basicConfig(level=logging.INFO)
import os
import math
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from pathlib import Path
from os.path import dirname, realpath, join

from utils_extract_from_corpora import extract_logic_language, extract_surface_language, extract_final_scripts
from utils_extract_from_corpora import extract_surface_logic_utterances, filter_region_graph_mapping, match_logical_surface_forms
from utils_extract_from_corpora import write_logic_to_surface, write_surface, write_sandwich

from utils_extract_from_corpora import increase_the_corpus
from utils_permutation_prompt import apply_permutations
from config import configs

'''
EXTRACT LANGUAGE FROM CORPORA FUNCTION
'''
def extract_mappings(configs, ids):

    logic_scripts = []

    surface_logic_mapping = []
    all_entities_map = {}        

    for vg_id, store_id in ids:

        logging.info(f"vg_id={vg_id}, store_id={store_id}")

        # 1. Extract surface logic mapping with new_situation_id=None
        entity_properties_map, _, entity_ids, entities_map = extract_logic_language(
            file_path=ideal_language_path,
            situation_id=vg_id,
            new_situation_id=None,
            limited=configs['limited'],
            limited_max_utterances=configs['limited_max_utterances'],
            non_included_ids=None
        )

        # In case ID was not found
        if not entity_properties_map:
            continue

        region_graph_mapping, ideallanguage_not_corresponding_ids = filter_region_graph_mapping(entity_ids, surface_logic_utterances)
        if len(ideallanguage_not_corresponding_ids) > 0:
            logging.info(f'Found {len(ideallanguage_not_corresponding_ids)} ids which are in ideallanguage but not in region_graph')
        mapping = match_logical_surface_forms(region_graph_mapping, entity_properties_map)

        surface_logic_mapping.append(mapping)
        all_entities_map[store_id] = entities_map

        # 2. Extract logic scripts with new_situation_id
        logic_script_output = extract_logic_language(
            file_path=ideal_language_path,
            situation_id=vg_id,
            new_situation_id=store_id,
            limited=configs['limited'],
            limited_max_utterances=configs['limited_max_utterances'],
            non_included_ids=ideallanguage_not_corresponding_ids
        )

        if logic_script_output:
            logic_scripts.extend(logic_script_output)
        
    logging.info(f"{len(configs['ids'])} situation have been saved")

    return logic_scripts, surface_logic_mapping, all_entities_map

'''
EXTRACT MAPPINGS BASED ON PARAMETERS
'''

def extract_data(surface_logic_utterances, configs, fold="training"):

    if configs['test_mode']:
        ids = configs['ids'][:test_max_situations]

    n = int(len(configs['ids']) * configs['train_test_ratio'])
    if fold == "training":
        ids = configs['ids'][:n]
    else:
        n = len(configs['ids']) - n
        ids = configs['ids'][-n:]

    logging.info(f'ORIGINAL IDS {ids}')

    logic_scripts, surface_logic_mapping, all_entities_map = extract_mappings(configs, ids)
    write_to_files(logic_scripts, surface_logic_mapping, configs, fold=fold, augmented=False)
    # logging.info(f'ALL ENTITIES MAP ORIGINAL {all_entities_map}')

    if configs['extend_corpus']:

        all_aug_logic_scripts, all_aug_surface_logic_mapping, all_aug_all_entities_map = extract_mappings(configs, ids)
        write_to_files(all_aug_logic_scripts, all_aug_surface_logic_mapping, configs, fold=fold, augmented=True)
        return original_logic_scripts, original_surface_logic_mapping, all_aug_logic_scripts, all_aug_surface_logic_mapping
    return logic_scripts, surface_logic_mapping


'''
WRITING TO FILES FUNCTION
'''

def write_to_files(logic_scripts, surface_logic_mapping, configs, fold='training', augmented=False):
    if not augmented:
        logging.info("Creating training files. No corpus extension required.")
        dir_path_original = join(parent_dir, "data", fold, "original")
        Path(dir_path_original).mkdir(parents=True, exist_ok=True)
        logging.info(dir_path_original)

        with open(join(dir_path_original, "original_logic_to_logic.txt"), 'w', encoding='utf-8') as fin:
            fin.write(''.join(logic_scripts))

        logic_to_surface = write_logic_to_surface(
            join(dir_path_original, "original_logic_to_surface.txt"), 
            surface_logic_mapping, plus_index=1, reverse=False)
        surface_to_logic = write_logic_to_surface(
            join(dir_path_original, "original_surface_to_logic.txt"),
            surface_logic_mapping, plus_index=1, reverse=True)
        surface_to_surface = write_surface(
            join(dir_path_original, "original_surface_to_surface.txt"),
            surface_logic_mapping, plus_index=1)
        sandwich = write_sandwich(
            join(dir_path_original, "original_sandwich.txt"),
            surface_logic_mapping, plus_index=1)

    else:
        logging.info("Creating training files with corpus extension.")
        dir_path_augmented = join(parent_dir, "data", fold, "augmented")
        Path(dir_path_augmented).mkdir(parents=True, exist_ok=True)

        if write_all_files:
            with open(join(dir_path_augmented, "augmented_logic_to_logic.txt"), 'w', encoding='utf-8') as fin:
                fin.write(''.join(logic_scripts))
        logic_to_surface = write_logic_to_surface(
            join(dir_path_augmented, "augmented_logic_to_surface.txt"),
            surface_logic_mapping, plus_index=1, reverse=False)
        surface_to_logic = write_logic_to_surface(
            join(dir_path_augmented, "augmented_surface_to_logic.txt"),
            surface_logic_mapping, plus_index=1, reverse=True)
        surface_to_surface = write_surface(
            join(dir_path_augmented, "augmented_surface_to_surface.txt"),
            surface_logic_mapping, plus_index=1)
        sandwich = write_sandwich(
            join(dir_path_augmented, "augmented_sandwich.txt"),
            surface_logic_mapping, plus_index=1)

    return ''.join(logic_scripts), logic_to_surface, surface_to_logic, surface_to_surface, sandwich

'''
CALLING FUNCTION
'''

if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(script_dir)
    ideal_language_path = join(parent_dir, "data", "ideallanguage.txt")

    # Extracts all (HUM utterance, BOT utterance) pairs from region_graph.
    surface_logic_utterances = extract_surface_logic_utterances(join(os.path.dirname(parent_dir), "dsc", "region_graphs.json.dsc"))

    logging.info(f"Chosen parameters:\n{configs}")

    if configs['extend_corpus']:
        logging.info("Augmenting corpus...")
        all_aug_situation_id = increase_the_corpus(configs)
        logging.info(f'IDS INCREASED {all_aug_situation_id}')
        config['ids'] = all_aug_situation_id
        original_logic_scripts, original_surface_logic_mapping, all_aug_logic_scripts, all_aug_surface_logic_mapping = \
            extract_data(ideallanguage, surface_logic_utterances, configs, fold="training")

        logic_scripts = original_logic_scripts + all_aug_logic_scripts
        surface_logic_mapping= original_surface_logic_mapping + all_aug_surface_logic_mapping

    else:
        logging.info("No corpus augmentation required...")
        logic_scripts, surface_logic_mapping = extract_data(surface_logic_utterances, configs, fold="training")
        logic_scripts, surface_logic_mapping = extract_data(surface_logic_utterances, configs, fold="testing")

    if configs['apply_permutations']:
        apply_permutations()
