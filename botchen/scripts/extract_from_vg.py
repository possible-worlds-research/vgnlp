# The **`./scripts/extract_from_vg.py`** script extracts data from the Visual Genome (VG) dataset
# and converts it into a conversational format suitable for chatbot training (Botchen).
# The extracted data is saved in a text file for further processing.

# **Usage:** ```python3 ./scripts/extract_from_vg.py```. 
# Takes  **`./data/ideallanguage.txt`** as input. Gives **`./data/extracted_scripts.txt'** as output. 
# The user can change the VG situation ID which to revert to Botchen format at the end of the file. 
# Now we selected 12 scripts, covering diverse topics.

import logging

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
        logging.info('AUGMENTIG FLAG PROCESS')

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

    logging.info(f'{len(ids_list)} situation have been stored')

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

         increase_corpus_flag = False,
         training_and_test_sets = False,
         min_referent_overlap_ratio=None,
         min_target_overlap_ratio=None,
         min_content_length=None,
         max_content_length=None,
         max_per_referent=None,
         train_split_ratio=None):

    if test_mode:
        ids = ids[:test_max_examples]

    logging.info(f'ORIGINAL IDS {ids}')

    logic_scripts, surface_logic_mapping, all_entities_map = processing_languages_mappings(
        ideallanguage_file_path=ideallanguage,
        ids_list=ids,
        limited=limited,
        limited_max_utterances=limited_max_utterances,
        matches=matches,
        augmenting_flag=False)
    logging.info(f'ALL ENTITIES MAP ORIGINAL {all_entities_map}')

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
        logging.info(f'IDS INCREASED {all_aug_situation_id}')

        if training_and_test_sets is True:
            train_aug_logic_scripts, train_aug_surface_logic_mapping, train_aug_all_entities_map = processing_languages_mappings(
                ideallanguage_file_path=ideallanguage,
                ids_list=training_aug_situation_id, # To do: Understand how to deal with training/testing/all
                limited=limited,
                limited_max_utterances=limited_max_utterances,
                matches=matches,
                augmenting_flag=increase_corpus_flag)
            logging.info(f'TRAINING NEW AUG_ALL_ENTITIES_MAP {train_aug_all_entities_map}')

            test_aug_logic_scripts, test_aug_surface_logic_mapping, test_aug_all_entities_map = processing_languages_mappings(
                ideallanguage_file_path=ideallanguage,
                ids_list=test_aug_situation_id, # To do: Understand how to deal with training/testing/all
                limited=limited,
                limited_max_utterances=limited_max_utterances,
                matches=matches,
                augmenting_flag=increase_corpus_flag)
            logging.info(f'TESTING NEW AUG_ALL_ENTITIES_MAP {test_aug_all_entities_map}')

            return logic_scripts, surface_logic_mapping, train_aug_logic_scripts, train_aug_surface_logic_mapping, test_aug_logic_scripts, test_aug_surface_logic_mapping

        else: # If no training and test set
            all_aug_logic_scripts, all_aug_surface_logic_mapping, all_aug_all_entities_map = processing_languages_mappings(
                ideallanguage_file_path=ideallanguage,
                ids_list=all_aug_situation_id, # To do: Understand how to deal with training/testing/all
                limited=limited,
                limited_max_utterances=limited_max_utterances,
                matches=matches,
                augmenting_flag=increase_corpus_flag)
            logging.info(f'TRAINING NEW AUG_ALL_ENTITIES_MAP {all_aug_all_entities_map}')

            return logic_scripts, surface_logic_mapping, all_aug_logic_scripts, all_aug_surface_logic_mapping

    if not increase_corpus_flag:
        return logic_scripts, surface_logic_mapping

'''
WRITING TO FILES FUNCTION
'''

def create_training_files(logic_scripts, surface_logic_mapping, increase_corpus_flag = False, training_and_test_sets=False, plus_index_testing=None):

    if increase_corpus_flag is False:
        with open('./data/training/original_logic.txt', 'w', encoding='utf-8') as file:
            file.write(''.join(logic_scripts))
        write_logic_to_surface('./data/training/original_logic_to_surface.txt', surface_logic_mapping, plus_index=1, reverse=False)
        write_logic_to_surface('./data/training/original_surface_to_logic.txt', surface_logic_mapping, plus_index=1, reverse=True)
        write_surface('./data/training/original_surface.txt', surface_logic_mapping, plus_index=1)
        write_sandwich('./data/training/original_sandwich.txt', surface_logic_mapping, plus_index=1)

    if increase_corpus_flag is True:

        if training_and_test_sets is False:
            with open('./data/training/augmented_logic.txt', 'w', encoding='utf-8') as file:
                file.write(''.join(logic_scripts))
            write_logic_to_surface('./data/training/augmented_logic_to_surface.txt', surface_logic_mapping, plus_index=1, reverse=False)
            write_logic_to_surface('./data/training/augmented_surface_to_logic.txt', surface_logic_mapping, plus_index=1, reverse=True)
            write_surface('./data/training/augmented_surface.txt', surface_logic_mapping, plus_index=1)
            write_sandwich('./data/training/augmented_sandwich.txt', surface_logic_mapping, plus_index=1)

        if training_and_test_sets is True:
            with open('./data/training/testing_augmented_logic.txt', 'w', encoding='utf-8') as file:
                file.write(''.join(logic_scripts))
            write_logic_to_surface('./data/training/testing_augmented_logic_to_surface.txt', surface_logic_mapping, plus_index=plus_index_testing, reverse=False)
            write_logic_to_surface('./data/training/testing_augmented_surface_to_logic.txt', surface_logic_mapping, plus_index=plus_index_testing, reverse=True)
            write_surface('./data/training/testing_augmented_surface.txt', surface_logic_mapping, plus_index=plus_index_testing)
            write_sandwich('./data/training/testing_augmented_sandwich.txt', surface_logic_mapping, plus_index=plus_index_testing)

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

    increase_corpus_flag = True
    training_and_test_sets=False
    write_files = True

    limited = True
    limited_max_utterances = 5
    test_mode = True
    test_max_examples = 4

    if increase_corpus_flag:

        min_referent_overlap_ratio=0.7 # FOCUSED ON REFERENT SITUATION Minimum proportion of referent entities that must appear in a target situation (i.e. we apply this to referent situations, e.g. *1 if the referent situation is as such)
        min_target_overlap_ratio=0.1 # FOCUSED ON TARGET SITUATION  Minimum proportion of target entities that must match referent entities (i.e. we apply this to all the *10 situations which we are finding similar to a referent situation *1)
        min_content_length=1000 # Minimum number of characters in a situation's content
        max_content_length=10000 # Maximum number of characters in a situation's content
        max_per_referent=5 # Maximum number of similar situations to extract per referent situation (e.g. we take *10* situations similar to situation 1, *10* to situation 2)
        train_split_ratio=0.7 # Percentage of training and testing sets

        '''
        ENDING DYNAMICAL PARAMETERS WHICH THE USER CAN CHANGE
        '''

        if training_and_test_sets is False:

            original_logic_scripts, original_surface_logic_mapping, \
            all_aug_logic_scripts, all_aug_surface_logic_mapping = \
                extract_languages(
                    ideallanguage,matches,ids,limited,limited_max_utterances,test_mode,test_max_examples, increase_corpus_flag, training_and_test_sets,
                    min_referent_overlap_ratio, min_target_overlap_ratio, min_content_length, max_content_length, max_per_referent, train_split_ratio)

            logic_scripts = original_logic_scripts + all_aug_logic_scripts
            surface_logic_mapping= original_surface_logic_mapping + all_aug_surface_logic_mapping

        if training_and_test_sets is True:

            original_logic_scripts, original_surface_logic_mapping, \
            train_aug_logic_scripts, train_aug_surface_logic_mapping, \
            test_aug_logic_scripts, test_aug_surface_logic_mapping = \
                extract_languages(
                ideallanguage,matches,ids,limited,limited_max_utterances,test_mode,test_max_examples, increase_corpus_flag, training_and_test_sets,
                min_referent_overlap_ratio, min_target_overlap_ratio, min_content_length, max_content_length, max_per_referent, train_split_ratio)

            logic_scripts = original_logic_scripts + train_aug_logic_scripts
            surface_logic_mapping= original_surface_logic_mapping + train_aug_surface_logic_mapping

            if write_files is True:
                # Adding this for the testing files
                create_training_files(logic_scripts=test_aug_logic_scripts,
                                      surface_logic_mapping=test_aug_surface_logic_mapping,
                                      increase_corpus_flag = increase_corpus_flag,
                                      training_and_test_sets = True,
                                      plus_index_testing=len(surface_logic_mapping)+1)

    else: # If not increase flag
        logic_scripts, surface_logic_mapping = extract_languages(
            ideallanguage="./data/ideallanguage.txt",
            matches=matches,
            ids = ids,
            limited=limited,
            limited_max_utterances=limited_max_utterances, # If limited, max of utterances/entities per situation
            test_mode=test_mode,
            test_max_examples=test_max_examples,  # If test mode true, max of situations extracted from total ids
            increase_corpus_flag = increase_corpus_flag
        )

    if write_files is True:
        create_training_files(logic_scripts=logic_scripts,
                              surface_logic_mapping=surface_logic_mapping,
                              increase_corpus_flag = increase_corpus_flag,
                              training_and_test_sets = False)