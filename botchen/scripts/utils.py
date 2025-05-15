# The **`./scripts/extract_from_vg.py`** script extracts data from the Visual Genome (VG) dataset
# and converts it into a conversational format suitable for chatbot training (Botchen).
# The extracted data is saved in a text file for further processing.

# **Usage:** ```python3 ./scripts/extract_from_vg.py```. 
# Takes  **`./data/ideallanguage.txt`** as input. Gives **`./data/extracted_scripts.txt'** as output. 
# The user can change the VG situation ID which to revert to Botchen format at the end of the file. 
# Now we selected 12 scripts, covering diverse topics.

import re
import random
import logging
import os
from collections import defaultdict
logging.basicConfig(level=logging.INFO)

''' 
LOGIC TO LOGIC
Function to extract a specific situation from the Visual Genome dataset and transform it into a conversational format.
Also, to extract entities and properties
Ideallanguage is the input file: './data/ideallanguage.txt'
'''
# file_path:str, situation_id:str, new_situation_id:Optional[str]=None, limited:bool=False
def extract_logic_language(file_path, situation_id, new_situation_id, limited, limited_max_utterances):
    logging.info('Logic-logic mapping process began')
    with open(file_path, 'r') as file:
        content = file.read()

    situation_pattern = rf"<situation id={situation_id}>(.*?)</situation>"
    match = re.search(situation_pattern, content, re.DOTALL)

    if not match:
        logging.info(f"No situation found with id={situation_id}")
        return None

    situation_body = match.group(1)

    # Maps: entity ID -> entity name / list of properties
    entity_id_to_name = {}
    entity_id_to_properties = {}

    # Extract entity IDs
    entity_id_regex = r'<entity id=(\d+)>'
    entity_ids_in_block = re.findall(entity_id_regex, situation_body)

    # Get all full entity blocks
    entity_block_regex = re.compile(r"<entity id=(\d+)>\s*(.*?)\s*</entity>", re.DOTALL)
    entity_blocks = list(entity_block_regex.finditer(situation_body))

    if limited:
        # For test mode: limit to first x entities
        entity_blocks = entity_blocks[:limited_max_utterances]
        entity_ids_in_block = [match.group(1) for match in entity_blocks]

    # FIRST PASS: Extract entity names and initialize property lists
    for entity_match in entity_blocks:
        entity_id = entity_match.group(1)
        entity_content_lines = entity_match.group(2).strip().split("\n")
        first_line = entity_content_lines[0].strip()

        entity_name_match = re.match(r"([\w\.]+)\(\d+\)", first_line)
        if entity_name_match:
            raw_entity_name = entity_name_match.group(1)
            normalized_entity_name = re.sub(r"\.n\.\d+", ".n", raw_entity_name)

            entity_id_to_name[entity_id] = normalized_entity_name
            entity_id_to_properties[entity_id] = []

    # SECOND PASS: Extract properties and between-entity relations
    for entity_match in entity_blocks:
        entity_id = entity_match.group(1)
        entity_content_lines = entity_match.group(2).strip().split("\n")

        for line in entity_content_lines[1:]:
            line = line.strip()
            prop_match = re.match(r"([\w|]+)\(([\d,]+)\)", line)
            if prop_match:
                relation_or_property, referenced_ids_str = prop_match.groups()
                referenced_ids = referenced_ids_str.split(",")

                # If it's a relation between entities, store the relationship.
                if len(referenced_ids) > 1:
                    source_id, target_id = referenced_ids
                    if source_id == entity_id and target_id in entity_id_to_name:
                        target_name = re.sub(r"\.n", "", entity_id_to_name[target_id])
                        entity_id_to_properties[entity_id].append(f"{relation_or_property}-{target_name}")
                    elif target_id == entity_id and source_id in entity_id_to_name:
                        source_name = re.sub(r"\.n", "", entity_id_to_name[source_id])
                        entity_id_to_properties[entity_id].append(f"{source_name}-{relation_or_property}")
                else:
                    clean_property_name = re.sub(r"\.n", "", relation_or_property)
                    entity_id_to_properties[entity_id].append(clean_property_name)

    # THIRD PASS: Generate new script blocks, if requested
    if new_situation_id:
        script_output = ''
        entity_list = list(entity_id_to_name.items())

        for i in range(len(entity_list)):
            curr_entity_id, curr_entity_name = entity_list[i]
            curr_props = entity_id_to_properties.get(curr_entity_id, [])
            cleaned_curr_props = [re.sub(r"\.n\.\d+", "", prop) for prop in curr_props]
            unique_curr_props = list(dict.fromkeys(cleaned_curr_props))

            script = f"<script.{new_situation_id} type=CONV>\n"
            script += f'<u speaker=HUM>({curr_entity_name} {" ".join(unique_curr_props)})</u>\n'

            if i + 1 < len(entity_list):
                next_entity_id, next_entity_name = entity_list[i + 1]
                next_props = entity_id_to_properties.get(next_entity_id, [])
                cleaned_next_props = [re.sub(r"\.n\.\d+", "", prop) for prop in next_props]
                unique_next_props = list(dict.fromkeys(cleaned_next_props))
                script += f'<u speaker=BOT>({next_entity_name} {" ".join(unique_next_props)})</u>\n'
            else:
                # Use the first utterance again as fallback
                first_entity_id, first_entity_name = entity_list[0]
                first_props = entity_id_to_properties.get(first_entity_id, [])
                cleaned_first_props = [re.sub(r"\.n\.\d+", "", prop) for prop in first_props]
                unique_first_props = list(dict.fromkeys(cleaned_first_props))
                script += f'<u speaker=BOT>({first_entity_name} {" ".join(unique_first_props)})</u>\n'
            script += f"</script.{new_situation_id}>\n\n"
            script_output += script
        logging.info('Logic-logic mapping process finished [script_output]')
        return script_output

    # DEFAULT: Return a dictionary of entity names and their properties
    if new_situation_id is None:
        entity_summary_map = {}
        for entity_id, name in entity_id_to_name.items():
            entity_summary_map[entity_id] = f"{name} {' '.join(entity_id_to_properties[entity_id])}"
        logging.info('Logic-logic mapping process finished [mapping only]')
        return entity_summary_map, situation_body, entity_ids_in_block, entity_id_to_name
# entity_summary_map: Dict[str, str], situation_body: str, entity_ids_in_block: List[str], entity_id_to_name: Dict[str, str]
# script_output: str

'''
SURFACE TO SURFACE
Extract the surface languages from the utterances for the situation
Region descriptions is the input file: './obs/region_descriptions.json.obs'
'''

# file_path:str, idx_to_extract:str, output_idx:str
def extract_surface_language(file_path, idx_to_extract, output_idx):
    logging.info('Surface-surface mapping process began')
    with open(file_path, 'r') as file:
        situation_pattern = rf"<a type=OBS idx={idx_to_extract}>(.*?)</a>"
        situation_data = re.search(situation_pattern, file.read(), re.DOTALL)
    if not situation_data:
        logging.info(f"No situation found with id={idx_to_extract}")
        return None
    situation_data = situation_data.group(1).strip()
    statements = re.findall(r'<e>(.*?)</e>', situation_data, re.DOTALL)
    new_file_total = ''
    for i in range(len(statements)):
        new_file = f"<script.{output_idx} type=CONV>\n"
        statement = statements[i]
        new_file += f"<u speaker=HUM>{statement.strip().lower()}</u>\n"
        if i + 1 < len(statements):
            next_statement = statements[i + 1]
            new_file += f"<u speaker=BOT>{next_statement.strip().lower()}</u>\n"
        else:
            old_statement = statements[0]
            new_file += f"<u speaker=BOT>{old_statement.strip().lower()}</u>\n"
        new_file += f"</script.{output_idx}>\n\n"
        new_file_total += new_file
    logging.info('Surface-surface mapping process finished')
    return new_file_total
# new_file_total:str

'''
LOGIC TO SURFACE AND INVERSE
Extract the matches between the object ids and the descriptions
Input file: './dsc/region_graphs.json.dsc'
'''

# Extracts all (HUM utterance, BOT utterance) pairs from region_graph.
# file_path:str
def extract_surface_logic_utterances(file_path):
    logging.info('Surface-logic mapping process began')
    pattern = re.compile(r'<a type=DSC idx=\d+>\s*<u speaker=HUM>(.*?)</u>\s*<u speaker=BOT>(.*?)</u>\s*</a>', re.DOTALL)
    results = []
    with open(file_path, 'r') as file:
        content = file.read()
        for match in pattern.finditer(content):
            hum_utt, bot_utt = match.groups()
            results.append((hum_utt.strip(), bot_utt.strip()))
    return results
# matches:list[tuple[str, str]]

# Filters and maps human entity IDs (from HUM utterance) to corresponding BOT utterance.
# valid_ids:set[str], dialogue_pairs:list[tuple[str, str]]
def filter_region_graph_mapping(valid_ids, dialogue_pairs):
    result = {}
    for hum_ids, bot_utterance in dialogue_pairs:
        cleaned_ids = hum_ids.strip('[]')
        hum_ids_list = cleaned_ids.split(',')  # In case there are multiple IDs
        valid_ids_found = [id.strip() for id in hum_ids_list if id.strip() in valid_ids]
        if valid_ids_found:
            result[', '.join(valid_ids_found)] = bot_utterance.strip()
    return result
# result:[str, str]

# Groups surface-form descriptions by shared logical forms.
# surface_map:dict[str, str], logical_map: dict[str, str]
def match_logical_surface_forms(surface_map, logical_map):
    new_map = {}
    for entity_ids, description in surface_map.items():
        entity_ids_list = entity_ids.split(", ")
        properties_list = []
        for entity_id in entity_ids_list:
            if entity_id in logical_map:
                properties_list.append(logical_map[entity_id])
        properties_str = "), (".join(sorted(set(properties_list)))
        key = f"({properties_str})"
        descriptions = [desc.strip().lower() for desc in description.split(" || ")]
        new_map.setdefault(key, set()).update(descriptions)
    logging.info('Surface-logic mapping process finished')
    return new_map
# new_map: dict[str, set[str]]

'''
Create training files
'''

def write_logic_to_surface(file_path, mapping, reverse=False):
    with open(file_path, "w", encoding="utf-8") as file:
        for situation in mapping:
            for hum_text, bot_texts in situation.items():
                for bot_text in bot_texts:
                    file.write(f'<a script.{(mapping.index(situation))+1} type=DSC>\n')
                    if reverse is True:
                        file.write(f'<u speaker=HUM>{bot_text}</u>\n')
                        file.write(f'<u speaker=BOT>{hum_text}</u>\n')
                    if reverse is False:
                        file.write(f'<u speaker=HUM>{hum_text}</u>\n')
                        file.write(f'<u speaker=BOT>{bot_text}</u>\n')
                    file.write(f'</a>\n\n')

def write_surface(file_path, mapping):
    with open(file_path, "w", encoding="utf-8") as file:
        for situation in mapping:
            situation_items = list(situation.items())
            total_bot_texts = []
            for hum_text, bot_text in situation_items:
                if isinstance(bot_text, (set, list)):
                    total_bot_texts.extend(bot_text)
            for i in range(len(total_bot_texts)):
                bot_text = total_bot_texts[i]
                file.write(f'<script.{(mapping.index(situation))+1} type=CONV>\n')
                file.write(f'<u speaker=HUM>{bot_text}</u>\n')
                if i + 1 < len(total_bot_texts):
                    new_bot_text = total_bot_texts[i + 1]
                    file.write(f'<u speaker=BOT>{new_bot_text}</u>\n')
                else: # If there is no pair, go back to the first utterance
                    old_bot_text = total_bot_texts[0]
                    file.write(f'<u speaker=BOT>{old_bot_text}</u>\n')
                file.write(f'</script.{(mapping.index(situation))+1}>\n\n')

def write_sandwich(file_path, mapping):
    with open(file_path, "w", encoding="utf-8") as file:
        for situation_idx, situation in enumerate(mapping):
            # Flattening the items such to repeat the surface forms attached to the logical forms in the list
            flattened_items = []
            for logical_form, surface_form_set in situation.items():
                surface_form_list = list(surface_form_set)
                for surface_form in surface_form_list:
                    flattened_items.append([logical_form, surface_form])
            for i in range(len(flattened_items)):
                file.write(f'<a script.{situation_idx + 1} type=SDW>\n')
                logical_form, surface_form = flattened_items[i]
                file.write(f"<u speaker=HUM>{logical_form}</u>\n")
                file.write(f"<u speaker=BOT>{surface_form}</u>\n")
                if i + 1 < len(flattened_items):
                    next_logical, next_surface = flattened_items[i + 1]
                    file.write(f"<u speaker=BOT>{next_surface}</u>\n")
                    file.write(f"<u speaker=BOT>{next_logical}</u>\n")
                else:
                    old_logical, old_surface = flattened_items[0]
                    file.write(f"<u speaker=BOT>{old_surface}</u>\n")
                    file.write(f"<u speaker=BOT>{old_logical}</u>\n")
                file.write(f'</a>\n\n')

'''
INCREASE DATA AMOUNT
'''

def increase_the_corpus(
        ideallanguage_path,
        all_entities_map,
        min_entity_overlap_ratio,
        min_target_overlap_ratio,
        min_content_length,
        max_content_length,
        max_per_referent,
        train_split_ratio
):
    logging.info('Increasing process began')
    with open(ideallanguage_path, 'r') as file:
        content = file.read()

    situation_pattern = r"<situation id=(.*?)>(.*?)</situation>"
    all_situations = re.findall(situation_pattern, content, re.DOTALL)

    all_situation_id = defaultdict(list)
    training_situation_id = defaultdict(list)
    test_situation_id = defaultdict(list)

    # Look at the situations from which we are augmenting
    for referent_id, referent_entity_map in all_entities_map.items():

        referent_entities = list(referent_entity_map.values())
        referent_set = set(referent_entities)
        logging.info(f'Referent Situation {referent_id} - Reference Entities: {referent_set}')
        matched_ids = []

        # For each situation in the corpus, iterate to see what we are extracting
        for target_id, situation_content in all_situations:
            entity_pattern = re.compile(r"<entity id=(\d+)>\s*(.*?)\s*</entity>", re.DOTALL)
            all_entities = list(entity_pattern.finditer(situation_content))

            # Look if there is  match and append
            extracted_names = []
            for match in all_entities:
                entity_lines = match.group(2).strip().split("\n")
                if not entity_lines:
                    continue
                first_line = entity_lines[0].strip()
                name_match = re.match(r"([\w\.]+)\(\d+\)", first_line)
                if name_match:
                    clean_name = re.sub(r"\.n\.\d+", ".n", name_match.group(1))
                    extracted_names.append(clean_name)
            target_set = set(extracted_names)

            # Look at the intersection and if there are all the conditions I want
            overlap = referent_set & target_set
            if referent_set and target_set:
                overlap_ratio_ref = len(overlap) / len(referent_set)
                overlap_ratio_tgt = len(overlap) / len(target_set)
                if (overlap_ratio_ref >= min_entity_overlap_ratio and
                    overlap_ratio_tgt >= min_target_overlap_ratio and
                    min_content_length <= len(situation_content) <= max_content_length):
                    matched_ids.append(target_id)

        logging.info(f"Referent {referent_id} matched with {len(matched_ids)} situations.")

        # If there is the possibility, divide between training and testing items. In any case, build a dictionary with all the items
        if matched_ids:
            selected = random.sample(matched_ids, min(len(matched_ids), max_per_referent))
            split_idx = int(len(selected) * (1 - train_split_ratio))
            test_situation_id[referent_id] = selected[:split_idx]
            training_situation_id[referent_id] = selected[split_idx:]
            all_situation_id[referent_id] = selected
        else:
            logging.info("We couldn't auch")

    logging.info("All situation ids:", dict(all_situation_id))
    print("Training situation ids:", dict(training_situation_id))
    print("Test situatios ids:", dict(test_situation_id))

    return dict(all_situation_id), dict(training_situation_id), dict(test_situation_id)