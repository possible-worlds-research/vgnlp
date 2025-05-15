# The **`./scripts/extract_from_vg.py`** script extracts data from the Visual Genome (VG) dataset
# and converts it into a conversational format suitable for chatbot training (Botchen).
# The extracted data is saved in a text file for further processing.

# **Usage:** ```python3 ./scripts/extract_from_vg.py```. 
# Takes  **`./data/ideallanguage.txt`** as input. Gives **`./data/extracted_scripts.txt'** as output. 
# The user can change the VG situation ID which to revert to Botchen format at the end of the file. 
# Now we selected 12 scripts, covering diverse topics.

import re
import os
import zipfile
import argparse
import json
import random
import logging

from collections import defaultdict
from utils import extract_logic_language, extract_surface_language
from utils import extract_surface_logic_utterances, filter_region_graph_mapping, match_logical_surface_forms
from utils import write_logic_to_surface, write_surface, write_sandwich
from utils import increase_the_corpus

'''
UTILS Functions description:

(A) extract_logic_language(file_path, situation_id, new_situation_id=None, limited=False) 
    - Inputs:
        * file_path: Path to the Ideallanguage file to extract from.
        * situation_id: The situation ID in Ideallanguage to extract utterances from.
        * new_situation_id (optional): If provided, remaps utterances to this new ID and formats output for training.
        * limited (bool): If True, limits extraction to limited_max_utterances utterances/entities for testing.
        * limited_max_utterances: str, max utterance for limited option
    - Returns:
        * If new_situation_id is provided:
            - Returns: script_output, script strings for training data, e.g.
                <script.2 type=CONV>
                    <u speaker=HUM>(apple.n red)</u>
                    <u speaker=BOT>(cat.n on-table)</u>
                </script.2>   
        * If not provided:
            - Returns: 
                - entity_summary_map: dict mapping numeric entity IDs to logic forms and properties.
                - situation_body: string containing situation content.
                - entity_ids_in_block: list of numeric entity IDs.
                - entity_id_to_name: list mapping entity IDs to their names.

(B) extract_surface_language(file_path, idx_to_extract, output_idx)
    - Inputs:
        * file_path: Path to the region_descriptions.json.obs file.
        * idx_to_extract: ID to extract utterances from.
        * output_idx: New assigned output ID.
    - Returns:
        * new_script: Content as a string for surface language, e.g.
            <script.2 type=CONV>
                <u speaker=HUM>the apple is red.</u>
                <u speaker=BOT>cat is on the table.</u>
            </script.2>    
    - Note:
        * Optional function; surface mapping can also be done more efficiently via region_graph file.

(C.1) extract_surface_logic_utterances(file_path)
    - Inputs:
        * file_path: Path to the region_graph file.
    - Returns:
        * matches: List of tuples pairing entity ID strings with surface sentences, e.g.
            [("[101]", "The apple is red."), ("[102, 103]", "The table supports the lamp.")]
(C.2) filter_region_graph_mapping(valid_ids, dialogue_pairs)
    - Inputs:
        * valid_ids: Set of entity IDs to include in mapping.
        * dialogue_pairs: List of dialogue pairs (a list of (HUM utterance, BOT utterance) tuples).
    - Returns:
        * result: Dict mapping entity ID strings to surface sentences, e.g.
            {"101": "The apple is red.", "102, 103": "The table supports the lamp."}
(C.3) match_logical_surface_forms(surface_map, logical_map)
    - Inputs:
        * surface_map: Mapping of entities/properties in surface form (from filter_region_graph_mapping).
        * logical_map: Mapping of entities/properties in logical form (from extract_logical_forms with new_situation_id=None).
    - Returns:
        * dict mapping logical forms to sets of corresponding surface expressions, e.g.
            {"(lamp.n), (on-table)": {"a lamp on a table"}, "(red)": {"a red apple"}}
'''

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

    if augmenting_flag is True:
        print('AAAAA', ids_list)
        iterable = [
            (vg_id, int(situation_list[-1]))
            for vg_id, situation_list in ids_list.items()
            if situation_list and str(situation_list[-1]).isdigit()
        ]
        print('ITERABLE', iterable)
    else:
        iterable = ids_list

    for i, (vg_id, store_id) in enumerate(iterable):
        logging.info("LOGIC: vg_id=%s, store_id=%s", vg_id, store_id)

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
        logging.info("SURFACE AND LOGIC-SURFACE MAPPING")
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
         test_max_examples=3,
         increase_corpus_flag = False):

    if test_mode:
        ids = ids[:test_max_examples]
    print('IDS NORMAL', ids)
    logic_scripts, surface_logic_mapping, all_entities_map = processing_languages_mappings(
        ideallanguage_file_path=ideallanguage,
        ids_list=ids,
        limited=limited,
        limited_max_utterances=limited_max_utterances,
        matches=matches,
        augmenting_flag=False)

    if increase_corpus_flag is True:
        all_aug_situation_id, training_aug_situation_id, test_aug_situation_id = increase_the_corpus(
            ideallanguage_path=ideallanguage,
            all_entities_map=all_entities_map,
            min_entity_overlap_ratio=0.8,
            min_target_overlap_ratio=0.1,
            min_content_length=1000,
            max_content_length=9000,
            max_per_referent=10,
            train_split_ratio=0.7
        )
        print('IDS INCREASED', all_aug_situation_id)
        aug_logic_scripts, aug_surface_logic_mapping, aug_all_entities_map = processing_languages_mappings(
            ideallanguage_file_path=ideallanguage,
            ids_list=all_aug_situation_id, # Here we could change based on training/testing or all
            limited=limited,
            limited_max_utterances=limited_max_utterances,
            matches=matches,
            augmenting_flag=increase_corpus_flag)

        return aug_logic_scripts, aug_surface_logic_mapping

    if not increase_corpus_flag:
        return logic_scripts, surface_logic_mapping

'''
WRITING TO FILES FUNCTION
'''

def create_training_files(logic_scripts, surface_logic_mapping, increase_corpus_flag = False):
    if not increase_corpus_flag:
        with open('./data/training/original_logic.txt', 'w', encoding='utf-8') as file:
            file.write(''.join(logic_scripts))
        write_logic_to_surface('./data/training/original_logic_to_surface.txt', surface_logic_mapping, reverse=False)
        write_logic_to_surface('./data/training/original_surface_to_logic.txt', surface_logic_mapping, reverse=True)
        write_surface('./data/training/original_surface.txt', surface_logic_mapping)
        write_sandwich('./data/training/original_sandwich.txt', surface_logic_mapping)

    if increase_corpus_flag:
        with open('./data/training/augmented_logic.txt', 'w', encoding='utf-8') as file:
            file.write(''.join(logic_scripts))
        write_logic_to_surface('./data/training/augmented_logic_to_surface.txt', surface_logic_mapping, reverse=False)
        write_logic_to_surface('./data/training/augmented_surface_to_logic.txt', surface_logic_mapping, reverse=True)
        write_surface('./data/training/augmented_surface.txt', surface_logic_mapping)
        write_sandwich('./data/training/augmented_sandwich.txt', surface_logic_mapping)

'''
CALLING FUNCTION
'''

# Extracts all (HUM utterance, BOT utterance) pairs from region_graph.
matches = extract_surface_logic_utterances("../../vgnlp2/dsc/region_graphs.json.dsc")

if __name__ == "__main__":

    # These are the mapping ids from the ideallangueg/visualGenome to the new ids
    ids = [(1, 1), (3, 2), (4, 3), (71, 4), (9, 5),
           (2410753, 6), (713137, 7), (2412620, 8), (2412211, 9),
           (186, 10), (2396154, 11), (2317468, 12)]

    increase_corpus_flag = True

    logic_scripts, surface_logic_mapping = extract_languages(
        ideallanguage="./data/ideallanguage.txt",
        matches=matches,
        ids = ids,

        limited=True,
        limited_max_utterances=5,

        test_mode=True,
        test_max_examples=2,

        increase_corpus_flag = increase_corpus_flag,
    )

    create_training_files(logic_scripts=logic_scripts,
                          surface_logic_mapping=surface_logic_mapping,
                          increase_corpus_flag = increase_corpus_flag)