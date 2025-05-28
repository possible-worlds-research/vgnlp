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
    - `increase_corpus_flag`: If `True`, adds similar situations to increase data. Similarity is measured in terms of common entities.
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

from utils_extract_from_corpora import extract_logic_language, extract_surface_language
from utils_extract_from_corpora import extract_surface_logic_utterances, filter_region_graph_mapping, match_logical_surface_forms
from utils_extract_from_corpora import write_logic_to_surface, write_surface, write_sandwich

from utils_extract_from_corpora import increase_the_corpus

from utils_permutation_prompt import get_conceptnet_hypernyms_synonyms, generate_random_substitutions
from utils_permutation_prompt import prompt_surface_logic, permutation_surface_logic, extract_situations
from utils_permutation_prompt import permutation_sandwich_logic_surface_transl, prompt_sandwich_logic_surface_transl

'''
EXTRACT LANGUAGE FROM CORPORA FUNCTION
'''

def processing_languages_mappings(ideallanguage_file_path,
                                  ids_list,
                                  limited,
                                  limited_max_utterances,
                                  matches,
                                  augmenting_flag=False):

    logic_scripts = []

    surface_logic_mapping = []
    all_entities_map = {}        

    for vg_id, store_id in ids_list:

        logging.info(f"vg_id={vg_id}, store_id={store_id}")
        # logging.info(f"LOGIC")

        # 1. Extract logic scripts with new_situation_id
        logic_script_output = extract_logic_language(
            file_path=ideallanguage_file_path,
            situation_id=vg_id,
            new_situation_id=store_id,
            limited=limited,
            limited_max_utterances=limited_max_utterances,
        )

        if logic_script_output:
            logic_scripts.extend(logic_script_output)

        # 2. Extract surface logic mapping with new_situation_id=None
        # logging.info("SURFACE AND LOGIC-SURFACE MAPPING")
        entity_properties_map, _, entity_ids, entities_map = extract_logic_language(
            file_path=ideallanguage_file_path,
            situation_id=vg_id,
            new_situation_id=None,
            limited=limited,
            limited_max_utterances=limited_max_utterances,
        )

        region_graph_mapping = filter_region_graph_mapping(entity_ids, matches)
        mapping = match_logical_surface_forms(region_graph_mapping, entity_properties_map)

        surface_logic_mapping.append(mapping)
        all_entities_map[store_id] = entities_map

    logging.info(f'{len(ids_list)} situation have been saved')

    return logic_scripts, surface_logic_mapping, all_entities_map

'''
EXTRACT MAPPINGS BASED ON PARAMETERS
'''

def extract_languages(ideallanguage,
         matches,
         ids,
         limited=False,
         limited_max_utterances=5,
         test_mode=False,
         test_max_situations=3,

         increase_corpus_flag = False,
         training_and_test_sets = False,
         min_referent_overlap_ratio=None,
         min_target_overlap_ratio=None,
         min_content_length=None,
         max_content_length=None,
         max_per_referent=None,
         train_split_ratio=None):

    if test_mode:
        ids = ids[:test_max_situations]

    # logging.info(f'ORIGINAL IDS {ids}')

    logic_scripts, surface_logic_mapping, all_entities_map = processing_languages_mappings(
        ideallanguage_file_path=ideallanguage,
        ids_list=ids,
        limited=limited,
        limited_max_utterances=limited_max_utterances,
        matches=matches,
        augmenting_flag=False)
    # logging.info(f'ALL ENTITIES MAP ORIGINAL {all_entities_map}')

    if increase_corpus_flag is True:
        all_aug_situation_id, training_aug_situation_id, test_aug_situation_id = increase_the_corpus(
            ideallanguage_path=ideallanguage,
            original_situation_ids=ids,
            all_entities_map=all_entities_map,
            min_referent_overlap_ratio=min_referent_overlap_ratio,
            min_target_overlap_ratio=min_target_overlap_ratio,
            min_content_length=min_content_length,
            max_content_length=max_content_length,
            max_per_referent=max_per_referent,
            train_split_ratio=train_split_ratio
        )
        # logging.info(f'IDS INCREASED {all_aug_situation_id}')

        if training_and_test_sets is True:
            train_aug_logic_scripts, train_aug_surface_logic_mapping, train_aug_all_entities_map = processing_languages_mappings(
                ideallanguage_file_path=ideallanguage,
                ids_list=training_aug_situation_id, # To do: Understand how to deal with training/testing/all
                limited=limited,
                limited_max_utterances=limited_max_utterances,
                matches=matches,
                augmenting_flag=increase_corpus_flag)
            # logging.info(f'TRAINING NEW AUG_ALL_ENTITIES_MAP {train_aug_all_entities_map}')

            test_aug_logic_scripts, test_aug_surface_logic_mapping, test_aug_all_entities_map = processing_languages_mappings(
                ideallanguage_file_path=ideallanguage,
                ids_list=test_aug_situation_id, # To do: Understand how to deal with training/testing/all
                limited=limited,
                limited_max_utterances=limited_max_utterances,
                matches=matches,
                augmenting_flag=increase_corpus_flag)
            # logging.info(f'TESTING NEW AUG_ALL_ENTITIES_MAP {test_aug_all_entities_map}')

            return logic_scripts, surface_logic_mapping, train_aug_logic_scripts, train_aug_surface_logic_mapping, test_aug_logic_scripts, test_aug_surface_logic_mapping

        else: # If no training and test set
            all_aug_logic_scripts, all_aug_surface_logic_mapping, all_aug_all_entities_map = processing_languages_mappings(
                ideallanguage_file_path=ideallanguage,
                ids_list=all_aug_situation_id, # To do: Understand how to deal with training/testing/all
                limited=limited,
                limited_max_utterances=limited_max_utterances,
                matches=matches,
                augmenting_flag=increase_corpus_flag)
            # logging.info(f'TRAINING NEW AUG_ALL_ENTITIES_MAP {all_aug_all_entities_map}')

            return logic_scripts, surface_logic_mapping, all_aug_logic_scripts, all_aug_surface_logic_mapping

    if not increase_corpus_flag:
        return logic_scripts, surface_logic_mapping

'''
WRITING TO FILES FUNCTION
'''

def create_training_files(logic_scripts, surface_logic_mapping, increase_corpus_flag = False, training_and_test_sets=False, plus_index_testing=None, write_all_files=False):

    if increase_corpus_flag is False:
        if write_all_files is True:
            os.makedirs(os.path.dirname('./data/training/original/'), exist_ok=True)
            with open('./data/training/original/original_logic_to_logic.txt', 'w', encoding='utf-8') as file:
                file.write(''.join(logic_scripts))
        logic_to_surface = write_logic_to_surface('./data/training/original/original_logic_to_surface.txt', surface_logic_mapping, plus_index=1, reverse=False, write_all_files=write_all_files)
        surface_to_logic = write_logic_to_surface('./data/training/original/original_surface_to_logic.txt', surface_logic_mapping, plus_index=1, reverse=True, write_all_files=write_all_files)
        surface_to_surface = write_surface('./data/training/original/original_surface_to_surface.txt', surface_logic_mapping, plus_index=1, write_all_files=write_all_files)
        sandwich = write_sandwich('./data/training/original/original_sandwich.txt', surface_logic_mapping, plus_index=1, write_all_files=write_all_files)

    if increase_corpus_flag is True:

        if training_and_test_sets is False:
            if permutation_flag is False and write_all_files is True:
                os.makedirs(os.path.dirname('./data/training/augmented/'), exist_ok=True)
                with open('./data/training/augmented/augmented_logic_to_logic.txt', 'w', encoding='utf-8') as file:
                    file.write(''.join(logic_scripts))
            logic_to_surface = write_logic_to_surface('./data/training/augmented/augmented_logic_to_surface.txt', surface_logic_mapping, plus_index=1, reverse=False, write_all_files=write_all_files)
            surface_to_logic = write_logic_to_surface('./data/training/augmented/augmented_surface_to_logic.txt', surface_logic_mapping, plus_index=1, reverse=True, write_all_files=write_all_files)
            surface_to_surface = write_surface('./data/training/augmented/augmented_surfaceto_surface.txt', surface_logic_mapping, plus_index=1, write_all_files=write_all_files)
            sandwich = write_sandwich('./data/training/augmented/augmented_sandwich.txt', surface_logic_mapping, plus_index=1, write_all_files=write_all_files)

        if training_and_test_sets is True:
            if permutation_flag is False and write_all_files is True:
                os.makedirs(os.path.dirname('./data/training/augmented/'), exist_ok=True)
                with open('./data/training/augmented/testing_augmented_logic_to_logic.txt', 'w', encoding='utf-8') as file:
                    file.write(''.join(logic_scripts))
            logic_to_surface = write_logic_to_surface('./data/testing/augmented/testing_augmented_logic_to_surface.txt', surface_logic_mapping, plus_index=plus_index_testing, reverse=False, write_all_files=write_all_files)
            surface_to_logic = write_logic_to_surface('./data/testing/augmented/testing_augmented_surface_to_logic.txt', surface_logic_mapping, plus_index=plus_index_testing, reverse=True, write_all_files=write_all_files)
            surface_to_surface = write_surface('./data/testing/augmented/testing_augmented_surfaceto_surface.txt', surface_logic_mapping, plus_index=plus_index_testing, write_all_files=write_all_files)
            sandwich = write_sandwich('./data/testing/augmented/testing_augmented_sandwich.txt', surface_logic_mapping, plus_index=plus_index_testing, write_all_files=write_all_files)

    return ''.join(logic_scripts), logic_to_surface, surface_to_logic, surface_to_surface, sandwich

'''
CALLING FUNCTION
'''

if __name__ == "__main__":

    # Extracts all (HUM utterance, BOT utterance) pairs from region_graph.
    matches = extract_surface_logic_utterances("../../vgnlp2/dsc/region_graphs.json.dsc")
    ideallanguage="./data/ideallanguage.txt"

    '''
    BEGINNING DYNAMICAL PARAMETERS WHICH THE USER CAN CHANGE
    '''

    # These are the mapping ids from the ideallangueg/visualGenome to the new ids
    ids = [(1, 1), (3, 2), (4, 3), (71, 4), (9, 5),
           (2410753, 6), (713137, 7), (2412620, 8), (2412211, 9),
           (186, 10), (2396154, 11), (2317468, 12)]

    substitution_terms_list = [
            'car','jacket','shirt', 'man','woman', 'tree','road', 'bicycle', 
            'gym_shoe','table', 'curtain', 'sofa', 'chair', 'picture', 'teddy', 
            'desk','jean','room', 'ceiling', 'shelf', 'picture', 'monitor', 'bottle', 
            'sunset', 'mouse', 'part','cup', 'egg', 'muffin', 'plate', 'tomato', 
            'sauce', 'tea', 'spoon','mouth', 'watch','giraffe', 'branch', 
            'neck', 'eye','basket', 'ginger', 'vegetable', 'bowl', 'cheese', 
            'chopstick','grass', 'elephant', 'trunk','suit', 'belt', 'hair', 'earring'
        ]

    increase_corpus_flag = True
    training_and_test_sets=True

    permutation_flag = True  # This applies the permutations and writes the files 

    write_all_files = True # This makes it write files of augmented and original

    limited = True
    limited_max_utterances = 4 # These make the situation be of x utterances

    test_mode = True
    test_max_situations = 2 # These make the x situations from which we extract

    logging.info(f"Chosen parameters:\nchosen ids: {ids}\nincrease_corpus_flag: {increase_corpus_flag}, training_and_test_sets: {training_and_test_sets}, permutation_flag: {permutation_flag}, write_all_files: {write_all_files}, limited: {limited}, limited_max_utterances: {limited_max_utterances}, test_mode: {test_mode}, test_max_situations: {test_max_situations}")
    if increase_corpus_flag:

        min_referent_overlap_ratio=0.7 # FOCUSED ON REFERENT SITUATION Minimum proportion of referent entities that must appear in a target situation (i.e. we apply this to referent situations, e.g. *1 if the referent situation is as such)
        min_target_overlap_ratio=0.1 # FOCUSED ON TARGET SITUATION  Minimum proportion of target entities that must match referent entities (i.e. we apply this to all the *10 situations which we are finding similar to a referent situation *1)
        min_content_length=1000 # Minimum number of characters in a situation's content
        max_content_length=10000 # Maximum number of characters in a situation's content
        max_per_referent=2 # Maximum number of similar situations to extract per referent situation (e.g. we take *10* situations similar to situation 1, *10* to situation 2)
        train_split_ratio=0.5 # Percentage of training and testing sets
        logging.info(f"min_referent_overlap_ratio {min_referent_overlap_ratio}, min_target_overlap_ratio {min_target_overlap_ratio}, min_content_length {min_content_length}, max_content_length {max_content_length}, max_per_referent {max_per_referent}, train_split_ratio {train_split_ratio}")
        
        '''
        ENDING DYNAMICAL PARAMETERS WHICH THE USER CAN CHANGE
        '''

        if training_and_test_sets is False:

            original_logic_scripts, original_surface_logic_mapping, \
            all_aug_logic_scripts, all_aug_surface_logic_mapping = \
                extract_languages(
                    ideallanguage,matches,ids,limited,limited_max_utterances,test_mode,test_max_situations, increase_corpus_flag, training_and_test_sets,
                    min_referent_overlap_ratio, min_target_overlap_ratio, min_content_length, max_content_length, max_per_referent, train_split_ratio)

            logic_scripts = original_logic_scripts + all_aug_logic_scripts
            surface_logic_mapping= original_surface_logic_mapping + all_aug_surface_logic_mapping


        if training_and_test_sets is True:

            original_logic_scripts, original_surface_logic_mapping, \
            train_aug_logic_scripts, train_aug_surface_logic_mapping, \
            test_aug_logic_scripts, test_aug_surface_logic_mapping = \
                extract_languages(
                ideallanguage,matches,ids,limited,limited_max_utterances,test_mode,test_max_situations, increase_corpus_flag, training_and_test_sets,
                min_referent_overlap_ratio, min_target_overlap_ratio, min_content_length, max_content_length, max_per_referent, train_split_ratio)

            logic_scripts =  original_logic_scripts + train_aug_logic_scripts
            surface_logic_mapping=  original_surface_logic_mapping + train_aug_surface_logic_mapping

            # Adding this for the testing files
            testing_logic_scripts, testing_logic_to_surface, testing_surface_to_logic, testing_surface_to_surface, testing_sandwich = create_training_files(logic_scripts=test_aug_logic_scripts,
                                  surface_logic_mapping=test_aug_surface_logic_mapping,
                                  increase_corpus_flag = increase_corpus_flag,
                                  training_and_test_sets = True,
                                  plus_index_testing=len(surface_logic_mapping)+1,
                                  write_all_files=write_all_files)

    else: # If not increase flag
        logic_scripts, surface_logic_mapping = extract_languages(
            ideallanguage="./data/ideallanguage.txt",
            matches=matches,
            ids = ids,
            limited=limited,
            limited_max_utterances=limited_max_utterances, # If limited, max of utterances/entities per situation
            test_mode=test_mode,
            test_max_situations=test_max_situations,  # If test mode true, max of situations extracted from total ids
            increase_corpus_flag = increase_corpus_flag
        )

    logic_to_logic_text, logic_to_surface_text, surface_to_logic_text, surface_to_surface_text, sandwich_text = create_training_files(logic_scripts=logic_scripts,
                          surface_logic_mapping=surface_logic_mapping,
                          increase_corpus_flag = increase_corpus_flag,
                          training_and_test_sets = False,
                          write_all_files=write_all_files)

    if permutation_flag:

        logging.info('PERMUTATION PROCESS BEGAN')

        substitution_dict = get_conceptnet_hypernyms_synonyms(substitution_terms_list)

        logging.info('Logic to logic permutation')
        permuted_logic_to_logic = permutation_surface_logic(extract_situations(logic_to_logic_text), substitution_dict)
        prompt_logic_to_logic = prompt_surface_logic(permuted_logic_to_logic)

        logging.info('Surface to surface permutation')
        permuted_surface_to_surface = permutation_surface_logic(extract_situations(surface_to_surface_text), substitution_dict)
        prompt_surface_to_surface = prompt_surface_logic(permuted_surface_to_surface)

        logging.info('Logic to surface permutation')
        permuted_logic_to_surface = permutation_sandwich_logic_surface_transl(logic_to_surface_text, substitution_dict)
        prompt_logic_to_surface = prompt_sandwich_logic_surface_transl(logic_to_surface_text)

        logging.info('Surface to logic permutation')
        permuted_surface_to_logic = permutation_sandwich_logic_surface_transl(surface_to_logic_text, substitution_dict)
        prompt_surface_to_logic = prompt_sandwich_logic_surface_transl(surface_to_logic_text)

        logging.info('Sandwich permutation')
        permuted_sandwich = permutation_sandwich_logic_surface_transl(sandwich_text, substitution_dict, sandwich_flag=1)
        prompt_sandwich = prompt_sandwich_logic_surface_transl(sandwich_text, sandwich_flag=1)

        for name in [
            "permuted_logic_to_logic",
            "permuted_surface_to_surface",
            "permuted_logic_to_surface",
            "permuted_surface_to_logic",
            "permuted_sandwich"]:
                content = eval(name)
                os.makedirs(os.path.dirname(f'./data/training/permuted_files/{name}.txt'), exist_ok=True)
                with open(f'./data/training/permuted_files/{name}.txt', "w", encoding="utf-8") as f:
                    f.write(content)

        for name in [
            "prompt_logic_to_logic", "prompt_surface_to_surface", "prompt_logic_to_surface", "prompt_surface_to_logic","prompt_sandwich"]:
                content = eval(name)
                os.makedirs(os.path.dirname(f'./data/training/prompt_files/{name}.txt'), exist_ok=True)
                with open(f'./data/training/prompt_files/{name}.txt', "w", encoding="utf-8") as f:
                    f.write(content)

        if training_and_test_sets is True:

            logging.info('TESTING PERMUTATIONS')
            logging.info('Logic to logic permutation')

            permuted_testing_logic_scripts = permutation_surface_logic(extract_situations(testing_logic_scripts), substitution_dict)
            prompt_testing_logic_scripts = prompt_surface_logic(permuted_testing_logic_scripts)

            logging.info('Surface to surface permutation')

            permuted_testing_surface_to_surface = permutation_surface_logic(extract_situations(testing_surface_to_surface), substitution_dict)
            prompt_testing_surface_to_surface = prompt_surface_logic(permuted_testing_surface_to_surface)

            logging.info('Logic to surface permutation')

            permuted_testing_logic_to_surface = permutation_sandwich_logic_surface_transl(testing_logic_to_surface, substitution_dict)
            prompt_testing_logic_to_surface = prompt_sandwich_logic_surface_transl(testing_logic_to_surface)

            logging.info('Surface to logic permutation')

            permuted_testing_surface_to_logic = permutation_sandwich_logic_surface_transl(testing_surface_to_logic, substitution_dict)
            prompt_testing_surface_to_logic = prompt_sandwich_logic_surface_transl(testing_surface_to_logic)

            logging.info('Sandwich permutation')

            permuted_testing_sandwich = permutation_sandwich_logic_surface_transl(testing_sandwich, substitution_dict, sandwich_flag=1)
            prompt_testing_sandwich = prompt_sandwich_logic_surface_transl(testing_sandwich, sandwich_flag=1)

            for name in [
                "permuted_testing_logic_scripts",
                "permuted_testing_surface_to_surface",
                "permuted_testing_logic_to_surface",
                "permuted_testing_surface_to_logic",
                "permuted_testing_sandwich"]:
                
                content = eval(name)
                os.makedirs(os.path.dirname(f'./data/testing/permuted_files/{name}.txt'), exist_ok=True)
                with open(f'./data/testing/permuted_files/{name}.txt', "w", encoding="utf-8") as f:
                    f.write(content)

            for name in [
                "prompt_testing_logic_scripts",
                "prompt_testing_surface_to_surface",
                "prompt_testing_logic_to_surface",
                "prompt_testing_surface_to_logic",
                "prompt_testing_sandwich"]:
                
                content = eval(name)
                os.makedirs(os.path.dirname(f'./data/testing/prompt_files/{name}.txt'), exist_ok=True)
                with open(f'./data/testing/prompt_files/{name}.txt', "w", encoding="utf-8") as f:
                    f.write(content)
