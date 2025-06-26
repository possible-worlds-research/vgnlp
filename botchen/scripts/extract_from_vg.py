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

from utils_extract_from_corpora import extract_logic_language, extract_surface_language
from utils_extract_from_corpora import extract_surface_logic_utterances, filter_region_graph_mapping, match_logical_surface_forms
from utils_extract_from_corpora import write_logic_to_surface, write_surface, write_sandwich

from utils_extract_from_corpora import increase_the_corpus

from utils_permutation_prompt import get_conceptnet_hypernyms_synonyms, generate_random_substitutions
from utils_permutation_prompt import prompt_surface_logic, permutation_surface_logic, extract_situations
from utils_permutation_prompt import permutation_sandwich_logic_surface_transl, prompt_sandwich_logic_surface_transl, extract_final_scripts
from config import configs

'''
EXTRACT LANGUAGE FROM CORPORA FUNCTION
'''
def processing_languages_mappings(configs):

    logic_scripts = []

    surface_logic_mapping = []
    all_entities_map = {}        

    for vg_id, store_id in configs['ids']:

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

def extract_languages(surface_logic_utterances, configs):

    if configs['test_mode']:
        ids = configs['ids'][:test_max_situations]

    logging.info(f'ORIGINAL IDS {configs['ids']}')

    logic_scripts, surface_logic_mapping, all_entities_map = processing_languages_mappings(configs)
    # logging.info(f'ALL ENTITIES MAP ORIGINAL {all_entities_map}')

    if configs['extend_corpus']:
        all_aug_situation_id = increase_the_corpus(configs)
        # logging.info(f'IDS INCREASED {all_aug_situation_id}')

        all_aug_logic_scripts, all_aug_surface_logic_mapping, all_aug_all_entities_map = processing_languages_mappings(configs)
        return logic_scripts, surface_logic_mapping, all_aug_logic_scripts, all_aug_surface_logic_mapping

    else:
        return logic_scripts, surface_logic_mapping

'''
WRITING TO FILES FUNCTION
'''

def create_training_files(logic_scripts, surface_logic_mapping, configs):
    write_all_files = configs['write_all_files']

    if not configs['extend_corpus']:
        logging.info("Creating training files. No corpus extension required.")
        dir_path_original = join(parent_dir, "data", "training", "original")
        Path(dir_path_original).mkdir(parents=True, exist_ok=True)
        logging.info(dir_path_original)

        if write_all_files:
            with open(join(dir_path_original, "original_logic_to_logic.txt"), 'w', encoding='utf-8') as fin:
                fin.write(''.join(logic_scripts))

        logic_to_surface = write_logic_to_surface(
            join(dir_path_original, "original_logic_to_surface.txt"), 
            surface_logic_mapping, plus_index=1, reverse=False, write_all_files=write_all_files)
        surface_to_logic = write_logic_to_surface(
            join(dir_path_original, "original_surface_to_logic.txt"),
            surface_logic_mapping, plus_index=1, reverse=True, write_all_files=write_all_files)
        surface_to_surface = write_surface(
            join(dir_path_original, "original_surface_to_surface.txt"),
            surface_logic_mapping, plus_index=1, write_all_files=write_all_files)
        sandwich = write_sandwich(
            join(dir_path_original, "original_sandwich.txt"),
            surface_logic_mapping, plus_index=1, write_all_files=write_all_files)

    else:
        logging.info("Creating training files with corpus extension.")
        dir_path_augmented = join(parent_dir, "data", "training", "augmented")
        Path(dir_path_augmented).mkdir(parents=True, exist_ok=True)

        if write_all_files:
            with open(join(dir_path_augmented, "augmented_logic_to_logic.txt"), 'w', encoding='utf-8') as fin:
                fin.write(''.join(logic_scripts))
        logic_to_surface = write_logic_to_surface(
            join(dir_path_augmented, "augmented_logic_to_surface.txt"),
            surface_logic_mapping, plus_index=1, reverse=False, write_all_files=write_all_files)
        surface_to_logic = write_logic_to_surface(
            join(dir_path_augmented, "augmented_surface_to_logic.txt"),
            surface_logic_mapping, plus_index=1, reverse=True, write_all_files=write_all_files)
        surface_to_surface = write_surface(
            join(dir_path_augmented, "augmented_surface_to_surface.txt"),
            surface_logic_mapping, plus_index=1, write_all_files=write_all_files)
        sandwich = write_sandwich(
            join(dir_path_augmented, "augmented_sandwich.txt"),
            surface_logic_mapping, plus_index=1, write_all_files=write_all_files)

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
        original_logic_scripts, original_surface_logic_mapping, all_aug_logic_scripts, all_aug_surface_logic_mapping = \
            extract_languages(ideallanguage, surface_logic_utterances, configs)

        logic_scripts = original_logic_scripts + all_aug_logic_scripts
        surface_logic_mapping= original_surface_logic_mapping + all_aug_surface_logic_mapping

    else:
        logging.info("No corpus augmentation required...")
        logic_scripts, surface_logic_mapping = extract_languages(surface_logic_utterances, configs)

    logic_to_logic_text, logic_to_surface_text, surface_to_logic_text, surface_to_surface_text, sandwich_text = create_training_files(logic_scripts, surface_logic_mapping, configs)

    if configs['permutation_flag']:

        logging.info('Beginning of permutation process')

        substitution_dict = get_conceptnet_hypernyms_synonyms(substitution_terms_list)

        # logging.info('Logic to logic permutation')
        permuted_logic_to_logic = permutation_surface_logic(extract_situations(logic_to_logic_text), substitution_dict)
        prompt_logic_to_logic = prompt_surface_logic(permuted_logic_to_logic)

        # logging.info('Surface to surface permutation')
        permuted_surface_to_surface = permutation_surface_logic(extract_situations(surface_to_surface_text), substitution_dict)
        prompt_surface_to_surface = prompt_surface_logic(permuted_surface_to_surface)

        # logging.info('Logic to surface permutation')
        permuted_logic_to_surface = permutation_sandwich_logic_surface_transl(logic_to_surface_text, substitution_dict)
        prompt_logic_to_surface = prompt_sandwich_logic_surface_transl(logic_to_surface_text)

        # logging.info('Surface to logic permutation')
        permuted_surface_to_logic = permutation_sandwich_logic_surface_transl(surface_to_logic_text, substitution_dict)
        prompt_surface_to_logic = prompt_sandwich_logic_surface_transl(surface_to_logic_text)

        # logging.info('Sandwich permutation')
        permuted_sandwich = permutation_sandwich_logic_surface_transl(sandwich_text, substitution_dict, sandwich_flag=1)
        prompt_sandwich = prompt_sandwich_logic_surface_transl(sandwich_text, sandwich_flag=1)

        for name in ["permuted_logic_to_logic","permuted_surface_to_surface","permuted_logic_to_surface","permuted_surface_to_logic","permuted_sandwich"]:
                content = eval(name)
                os.makedirs(os.path.dirname(
                    join(parent_dir, "data", "training", "permuted_files", f"{name}.txt")), 
                    exist_ok=True)

                with open(
                    join(parent_dir, "data", "training", "permuted_files", f"{name}.txt"),
                    "w", encoding="utf-8") as f:
                    f.write(content)

                total_tokens = len(word_tokenize(content))
                logging.info(f'{name} has {total_tokens} tokens')

        for name in [
            "prompt_logic_to_logic", "prompt_surface_to_surface", "prompt_logic_to_surface", "prompt_surface_to_logic","prompt_sandwich"]:
                content = eval(name)
                os.makedirs(os.path.dirname(
                    join(parent_dir, "data", "training", "prompt_files", f"{name}.txt")), 
                    exist_ok=True)
                with open(
                    join(parent_dir, "data", "training", "prompt_files", f"{name}.txt"),
                    "w", encoding="utf-8") as f:
                    f.write(content)
                total_tokens = len(word_tokenize(content))
                logging.info(f'{name} has {total_tokens} tokens')

        if training_and_test_sets is True:

            logging.info('Training and testing split began')

            for name in [
                "prompt_logic_to_logic", "prompt_surface_to_surface", "prompt_logic_to_surface", "prompt_surface_to_logic", "prompt_sandwich"]:
                    
                    specific_dir = join(parent_dir, "data", "training", "prompt_files")

                    with open(join(specific_dir, f"{name}.txt"),"r", encoding="utf-8") as original_file:
                        content = original_file.read()

                    _, training_scripts, testing_scripts = extract_final_scripts(content, train_split_ratio)

                    # Training data
                    with open(join(specific_dir, f"{name}.txt"), "w", encoding="utf-8") as training_file:
                        training_file.write("\n\n".join(script for script in training_scripts))

                    # Testing data
                    os.makedirs(os.path.dirname(join(parent_dir, "data", "testing", f"testing_{name}.txt")), exist_ok=True)

                    with open(join(parent_dir, "data", "testing", f"testing_{name}.txt"), "w", encoding="utf-8") as testing_file:
                        testing_file.write("\n\n".join(script for script in testing_scripts))

            for name in [
                "permuted_logic_to_logic", "permuted_surface_to_surface", "permuted_logic_to_surface", "permuted_surface_to_logic", "permuted_sandwich"]:
                    
                    specific_dir = join(parent_dir, "data", "training", "permuted_files")

                    with open(join(specific_dir, f"{name}.txt"),"r", encoding="utf-8") as original_file:
                        content = original_file.read()

                    _, training_scripts, testing_scripts = extract_final_scripts(content, train_split_ratio)

                    # Training data
                    with open(join(specific_dir, f"{name}.txt"), "w", encoding="utf-8") as training_file:
                        training_file.write("\n\n".join(script for script in training_scripts))

                    # Testing data
                    os.makedirs(os.path.dirname(join(parent_dir, "data", "testing", f"testing_{name}.txt")), exist_ok=True)

                    with open(join(parent_dir, "data", "testing", f"testing_{name}.txt"), "w", encoding="utf-8") as testing_file:
                        testing_file.write("\n\n".join(script for script in testing_scripts))
