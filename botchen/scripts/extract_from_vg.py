# The **`./scripts/extract_from_vg.py`** script extracts data from the Visual Genome (VG) dataset
# and converts it into a conversational format suitable for chatbot training (Botchen).
# The extracted data is saved in a text file for further processing.

import logging
import os

from utils_extract_from_corpora import extract_logic_language, extract_surface_language
from utils_extract_from_corpora import extract_surface_logic_utterances, filter_region_graph_mapping, match_logical_surface_forms
from utils_extract_from_corpora import write_logic_to_surface, write_surface, write_sandwich

from utils_extract_from_corpora import increase_the_corpus

from utils_permutation_prompt import extract_entities_properties, prompt_surface_logic, permutation_surface_logic, extract_situations
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

def create_training_files(logic_scripts, surface_logic_mapping, increase_corpus_flag = False, training_and_test_sets=False, plus_index_testing=None, write_files=False):

    if increase_corpus_flag is False:
        os.makedirs(os.path.dirname('./data/training/original/'), exist_ok=True)
        with open('./data/training/original/original_logic.txt', 'w', encoding='utf-8') as file:
            file.write(''.join(logic_scripts))
        logic_to_surface = write_logic_to_surface('./data/training/original/original_logic_to_surface.txt', surface_logic_mapping, plus_index=1, reverse=False, write_files=write_files)
        surface_to_logic = write_logic_to_surface('./data/training/original/original_surface_to_logic.txt', surface_logic_mapping, plus_index=1, reverse=True, write_files=write_files)
        surface_to_surface = write_surface('./data/training/original/original_surface.txt', surface_logic_mapping, plus_index=1, write_files=write_files)
        sandwich = write_sandwich('./data/training/original/original_sandwich.txt', surface_logic_mapping, plus_index=1, write_files=write_files)

    if increase_corpus_flag is True:

        if training_and_test_sets is False:
            os.makedirs(os.path.dirname('./data/training/augmented/'), exist_ok=True)
            with open('./data/training/augmented/augmented_logic.txt', 'w', encoding='utf-8') as file:
                file.write(''.join(logic_scripts))
            logic_to_surface = write_logic_to_surface('./data/training/augmented/augmented_logic_to_surface.txt', surface_logic_mapping, plus_index=1, reverse=False, write_files=write_files)
            surface_to_logic = write_logic_to_surface('./data/training/augmented/augmented_surface_to_logic.txt', surface_logic_mapping, plus_index=1, reverse=True, write_files=write_files)
            surface_to_surface = write_surface('./data/training/augmented/augmented_surface.txt', surface_logic_mapping, plus_index=1, write_files=write_files)
            sandwich = write_sandwich('./data/training/augmented/augmented_sandwich.txt', surface_logic_mapping, plus_index=1, write_files=write_files)

        if training_and_test_sets is True:
            os.makedirs(os.path.dirname('./data/training/augmented/'), exist_ok=True)
            with open('./data/training/augmented/testing_augmented_logic.txt', 'w', encoding='utf-8') as file:
                file.write(''.join(logic_scripts))
            logic_to_surface = write_logic_to_surface('./data/training/augmented/testing_augmented_logic_to_surface.txt', surface_logic_mapping, plus_index=plus_index_testing, reverse=False, write_files=write_files)
            surface_to_logic = write_logic_to_surface('./data/training/augmented/testing_augmented_surface_to_logic.txt', surface_logic_mapping, plus_index=plus_index_testing, reverse=True, write_files=write_files)
            surface_to_surface = write_surface('./data/training/augmented/testing_augmented_surface.txt', surface_logic_mapping, plus_index=plus_index_testing, write_files=write_files)
            sandwich = write_sandwich('./data/training/augmented/testing_augmented_sandwich.txt', surface_logic_mapping, plus_index=plus_index_testing, write_files=write_files)

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
    # To do: Make these parameters updatable by terminal 

    # These are the mapping ids from the ideallangueg/visualGenome to the new ids
    ids = [(1, 1), (3, 2), (4, 3), (71, 4), (9, 5),
           (2410753, 6), (713137, 7), (2412620, 8), (2412211, 9),
           (186, 10), (2396154, 11), (2317468, 12)]

    increase_corpus_flag = False
    training_and_test_sets=False

    write_files = True # This makes it write files of augmented and original
    permutation_flag = True  # This applies the permutations and writes the files 

    limited = True
    limited_max_utterances = 5 # These make the situation be of x utterances

    test_mode = True
    test_max_examples = 4 # These make the situations from which we extract x number

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

            # Adding this for the testing files
            testing_logic_scripts, testing_logic_to_surface, testing_surface_to_logic, testing_surface_to_surface, testing_sandwich = create_training_files(logic_scripts=test_aug_logic_scripts,
                                  surface_logic_mapping=test_aug_surface_logic_mapping,
                                  increase_corpus_flag = increase_corpus_flag,
                                  training_and_test_sets = True,
                                  plus_index_testing=len(surface_logic_mapping)+1,
                                  write_files=write_files)

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

    logic_to_logic_text, logic_to_surface_text, surface_to_logic_text, surface_to_surface_text, sandwich_text = create_training_files(logic_scripts=logic_scripts,
                          surface_logic_mapping=surface_logic_mapping,
                          increase_corpus_flag = increase_corpus_flag,
                          training_and_test_sets = False,
                          write_files=write_files)

    if permutation_flag:

        # To do: This should be automatized with WordNet, also now it does not apply to all when we increase the corpus (because the logic is based on the situations)
        substitutions_per_situation = {
            1: [('', ''), ('car', 'vehicle'), ('jacket', 'raincoat'), ('shirt', 'sweater'),
                ('building', 'house'), ('wall', 'separation'), ('man', 'woman'),
                ('spectacles', 'sunglasses'), ('tree', 'plant')],
            2: [('', ''), ('road', 'street'), ('building', 'house'), ('man', 'woman'),
                ('car', 'scooter'), ('tree', 'plant'), ('light', 'lamp'),
                ('bicycle', 'motorcycle'), ('gym_shoe', 'boot')],
            3: [('', ''), ('table', 'support'), ('curtain', 'furniture'), ('sofa', 'armchair'),
                ('chair', 'seat'), ('picture', 'illustration'), ('teddy', 'puppet'),
                ('carpet', 'moquette'), ('desk', 'bureau')],
            4: [('', ''), ('woman', 'man'), ('box', 'container'), ('jean', 'skirt'), ('floor', 'pavement'),
                ('shirt', 'sweater'), ('table', 'desk'), ('tape', 'object'), ('desktop', 'device'),],
            5: [('', ''), ('room', 'bedroom'), ('light', 'lamp'), ('ceiling', 'roof'),
                ('desk', 'bureau'), ('shelf', 'ledge'), ('picture', 'poster'), ('monitor', 'screen'),
                ('bottle', 'container')],
            6: [('', ''), ('chair', 'couch'),('desk', 'table'),('picture', 'image'),
                ('sunset', 'element'),('lamp', 'light'),('mouse', 'device'),('monitor', 'screen'),('part', 'portion')],
            7: [('', ''), ('cup', 'mug'),('egg', 'cheese'),('muffin', 'pastry'),
                ('plate', 'dish'),('tomato', 'vegetable'),('sauce', 'condiment'),('tea', 'beverage'),('spoon', 'utensil')],
            8: [('', ''), ('chair', 'couch'),('man', 'woman'),('mouth', 'face'),
                ('button', 'piece'),('spectacles', 'glasses'),('light', 'lamp'),('shirt', 'dress'),('watch', 'accessory')],
            9: [('', ''), ('giraffe', 'animal'),('green_park', 'area'),('leaf', 'plant'),
                ('tree', 'plant'),('mouth', 'body'),('branch', 'limb'),('neck', 'body'),('eye', 'body')],
            10: [('', ''),('plate', 'dish'),('basket', 'container'),('tray', 'platter'),
                 ('ginger', 'bulb'),('vegetable', 'food'),('bowl', 'container'),('cheese', 'food'),('chopstick', 'object')],
            11: [('', ''), ('tree', 'plant'),('grass', 'plant'),('elephant', 'animal'),
                 ('back', 'body'),('trunk', 'body'),('ear', 'body'),('leaf', 'plant'),('tail', 'limb')],
            12: [('', ''), ('man', 'person'),('man', 'woman'),('suit', 'sweater'),
                 ('belt', 'accessory'),('eye', 'face'),('building', 'house'),('hair', 'head'),('earring', 'jewelry')],
        }

        permuted_logic_to_logic, prompt_logic = permutation_surface_logic(extract_situations(logic_to_logic_text), substitutions_per_situation,  surface_language=False)

        permuted_surface_to_surface, prompt_surface = permutation_surface_logic(extract_situations(surface_to_surface_text), substitutions_per_situation,  surface_language=True)

        permuted_logic_to_surface = permutation_sandwich_logic_surface_transl(logic_to_surface_text, substitutions_per_situation)
        prompt_logic_to_surface = prompt_sandwich_logic_surface_transl(logic_to_surface_text)

        permuted_surface_to_logic = permutation_sandwich_logic_surface_transl(surface_to_logic_text, substitutions_per_situation)
        prompt_surface_to_logic = prompt_sandwich_logic_surface_transl(surface_to_logic_text)

        permuted_sandwich = permutation_sandwich_logic_surface_transl(sandwich_text, substitutions_per_situation, sandwich_flag=1)
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
            "prompt_logic", "prompt_surface", "prompt_logic_to_surface", "prompt_surface_to_logic","prompt_sandwich"]:
                content = eval(name)
                os.makedirs(os.path.dirname(f'./data/training/prompt_files/{name}.txt'), exist_ok=True)
                with open(f'./data/training/prompt_files/{name}.txt', "w", encoding="utf-8") as f:
                    f.write(content)