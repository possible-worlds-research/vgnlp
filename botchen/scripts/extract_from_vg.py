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

from collections import defaultdict

################## LOGIC TO LOGIC

# Function to extract a specific situation from the Visual Genome dataset and transform it into a conversational format.
# Also, to extract entities and properties
# Ideallanguage is the input file: './data/ideallanguage.txt'

def extract_logical_forms(file_path, situation_id, new_situation_id=None, limited=False):
    with open(file_path, 'r') as file:
        content = file.read()
    # Patterns I will need to extract
    entity_id_pattern = r'<entity id=(\d+)>'
    entity_pattern = re.compile(r"<entity id=(\d+)>\s*(.*?)\s*</entity>", re.DOTALL)
    situation_pattern = rf"<situation id={situation_id}>(.*?)</situation>"
    # Extract the situation block by matching its ID.
    match = re.search(situation_pattern, content, re.DOTALL)
    if match:
        situation_content = match.group(1)
        # Dictionaries to store the entity details: entity names and their properties.
        entity_map = {}  # Maps entity ID -> entity name
        properties_map = {}  # Maps entity ID -> list of properties
        # Store the numerical id of the entities
        entity_matches = re.findall(entity_id_pattern, situation_content)
        # FIRST PASS: Extract and store each entity's ID, name, and initialize its properties.
        all_entities = list(entity_pattern.finditer(situation_content))
        if limited:
            # Limited training data to test, 6 utterances for situation
            all_entities = all_entities[:6]
            entity_matches = [match.group(1) for match in all_entities]
        for entity_match in all_entities:
            entity_id = entity_match.group(1)
            entity_lines = entity_match.group(2).strip().split("\n")
            # Extract the entity name from the first line.
            first_line = entity_lines[0].strip()
            entity_name_match = re.match(r"([\w\.]+)\(\d+\)", first_line)
            if entity_name_match:
                entity_name = entity_name_match.group(1)
                # Clean up entity name by removing numeric suffix (e.g., .n.xx becomes .n)
                clean_entity_name = re.sub(r"\.n\.\d+", ".n", entity_name)
                entity_map[entity_id] = clean_entity_name
                properties_map[entity_id] = []
        # SECOND PASS: Extract properties and relations between entities.
        for entity_match in all_entities:
            entity_id = entity_match.group(1)
            entity_lines = entity_match.group(2).strip().split("\n")
            for line in entity_lines[1:]:
                line = line.strip()
                prop_match = re.match(r"([\w|]+)\(([\d,]+)\)", line)
                if prop_match:
                    prop_name, prop_ids = prop_match.groups()
                    prop_ids = prop_ids.split(",")
                    # If it's a relation between entities, store the relationship.
                    if len(prop_ids) > 1:
                        ref_id_1, ref_id_2 = prop_ids
                        if ref_id_1 == entity_id and ref_id_2 in entity_map:
                            related_entity = entity_map[ref_id_2]
                            related_entity = re.sub(r"\.n", "", related_entity)
                            properties_map[entity_id].append(f"{prop_name}-{related_entity}")
                        elif ref_id_2 == entity_id and ref_id_1 in entity_map:
                            related_entity = entity_map[ref_id_1]
                            related_entity = re.sub(r"\.n", "", related_entity)
                            properties_map[entity_id].append(f"{related_entity}-{prop_name}")
                    else:
                        clean_prop = re.sub(r"\.n", "", prop_name)
                        properties_map[entity_id].append(clean_prop)
        # IF WE ARE CREATING NEW SITUATION IDS: THIRD PASS: Write the new files
        if new_situation_id:
            entity_items = list(entity_map.items())
            script_content_total = ''
            for i in range(len(entity_items)):
                entity_id, entity_name = entity_items[i]
                script_content = f"<script.{new_situation_id} type=CONV>\n"
                raw_props = properties_map.get(entity_id, [])
                cleaned_props = [re.sub(r"\.n\.\d+", "", prop) for prop in raw_props]
                unique_props = list(dict.fromkeys(cleaned_props))
                properties_str = " ".join(unique_props)
                script_content += f'<u speaker=HUM>({entity_name} {properties_str})</u>\n'
                # I do the increasing in this way such to have an utterances mapping: AABB BBCC etc.
                if i + 1 < len(entity_items):
                    new_entity_id, new_entity_name = entity_items[i+1]
                    new_raw_props = properties_map.get(new_entity_id, [])
                    new_cleaned_props = [re.sub(r"\.n\.\d+", "", prop) for prop in new_raw_props]
                    new_unique_props = list(dict.fromkeys(new_cleaned_props))
                    new_properties_str = " ".join(new_unique_props)
                    script_content += f'<u speaker=BOT>({new_entity_name} {new_properties_str})</u>\n'
                else:
                    # This is done in order to retrieve the very first utterance if there is no available pair
                    old_entity_id, old_entity_name = entity_items[0]
                    old_raw_props = properties_map.get(old_entity_id, [])
                    old_cleaned_props = [re.sub(r"\.n\.\d+", "", prop) for prop in old_raw_props]
                    old_unique_props = list(dict.fromkeys(old_cleaned_props))
                    old_properties_str = " ".join(old_unique_props)
                    script_content += f"<u speaker=BOT>({old_entity_name} {old_properties_str})</u>\n"
                script_content += f"</script.{new_situation_id}>\n\n"
                script_content_total += script_content
            return script_content_total
        # IF WE JUST WANT TO RETRIEVE THE ENTITIES OF THAT SITUATION
        if new_situation_id is None:
            unified_map = {}
            for entity_id, entity_name in entity_map.items():
                properties_str = " ".join(properties_map[entity_id])
                unified_map[entity_id] = f"{entity_map[entity_id]} {properties_str}"
            return unified_map, situation_content, entity_matches, entity_map
    if not match:
        print(f"No situation found with id={situation_id}")
        return None

############### SURFACE TO SURFACE

# Extract the surface languages from the utterances for the situation
# Region descriptions is the input file: './obs/region_descriptions.json.obs'

def extract_surface_language(file_path, idx_to_extract, output_idx):
    with open(file_path, 'r') as file:
        situation_pattern = rf"<a type=OBS idx={idx_to_extract}>(.*?)</a>"
        situation_data = re.search(situation_pattern, file.read(), re.DOTALL)
    if not situation_data:
        print(f"No situation found with id={idx_to_extract}")
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
    return new_file_total

############### LOGIC TO SURFACE AND INVERSE

# Extract the matches between the object ids and the descriptions
# Input file: './dsc/region_graphs.json.dsc'

def extract_matches(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    a_pattern = r'<a type=DSC idx=\d+>\s*<u speaker=HUM>(.*?)</u>\s*<u speaker=BOT>(.*?)</u>\s*</a>'
    matches = re.findall(a_pattern, content, re.DOTALL)
    return matches

# Extract from the mapping from extract_matches the ids of the situation I am extracting from
def extract_graph_mapping(valid_ids, matches):
    result = {}
    for hum_ids, bot_sentence in matches:
        hum_ids = hum_ids.strip('[]')
        hum_ids_list = hum_ids.split(',')  # In case there are multiple IDs
        valid_ids_found = [id.strip() for id in hum_ids_list if id.strip() in valid_ids]
        if valid_ids_found:
            result[', '.join(valid_ids_found)] = bot_sentence.strip()
    return result

# Extract the logical forms of the items and the properties and the sentences of description
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
        if key not in new_map:
            new_map[key] = set()
        new_map[key].update(descriptions)
    return new_map

############## INCREASE DATA AMOUNT

def increase_the_corpus(ideallanguage, all_entities_map):
    with open(ideallanguage, 'r') as file:
        content = file.read()

    situation_pattern = r"<situation id=(.*?)>(.*?)</situation>"
    situation_matches = re.findall(situation_pattern, content, re.DOTALL)

    all_situation_id = defaultdict(list)
    training_situation_id = defaultdict(list)
    test_situation_id = defaultdict(list)

    # Look at the situations from which we are augmenting
    for referent_situation_id, referent_entities_map in all_entities_map.items():

        referent_situation_entities = list(referent_entities_map.values())
        print(f'REFERENT SITUATION {referent_situation_id} - REFERENCE ENTITIES:', referent_situation_entities)

        referent_set = set(referent_situation_entities)
        # From each situation, extract the ids from which we will augment
        situation_match_ids = []

        # For each situation in the corpus, iterate to see what we are extracting
        for target_situation_id, target_situation_content in situation_matches:
            entity_pattern = re.compile(r"<entity id=(\d+)>\s*(.*?)\s*</entity>", re.DOTALL)
            all_entities = list(entity_pattern.finditer(target_situation_content))

            # Look if there is  match and append
            entity_names = []
            for entity_match in all_entities:
                entity_lines = entity_match.group(2).strip().split("\n")
                first_line = entity_lines[0].strip()
                entity_name_match = re.match(r"([\w\.]+)\(\d+\)", first_line)
                if entity_name_match:
                    entity_name = entity_name_match.group(1)
                    clean_entity_name = re.sub(r"\.n\.\d+", ".n", entity_name)
                    entity_names.append(clean_entity_name)
            target_set = set(entity_names)

            # Look at the intersection and if there are all the conditions I want
            intersection = referent_set & target_set
            if referent_set and target_set and \
                    len(intersection) / len(referent_set) >= 0.8 and \
                    len(intersection) / len(target_set) >= 0.1 and \
                    1000 <= len(target_situation_content) <= 9000:
                # print(f"Found a match between situation_id {target_situation_id} and referent situation_id {referent_situation_id}")
                # In case, append to specific referent_situation_id
                situation_match_ids.append(target_situation_id)

        print(f'Found {len(situation_match_ids)} items for referent {referent_situation_id}')

        # If there is the possibility, divide between training and testing items. In any case, build a dictionary with all the items
        if len(situation_match_ids) > 10:
            selected_items = random.sample(situation_match_ids, 10)
            training_situation_id[referent_situation_id].append(selected_items[3:])
            test_situation_id[referent_situation_id].append(selected_items[:3])
        else:
            selected_items=situation_match_ids
        all_situation_id[referent_situation_id].extend(selected_items)

    print('ALL SITUATIONS ID:', dict(all_situation_id))
    print('TEST SET:', dict(test_situation_id))
    print('TRAINING SET:', dict(training_situation_id))
    return dict(all_situation_id), dict(training_situation_id), dict(test_situation_id)

############# MAIN

# Main function to execute the extraction process for multiple situations and save the result.
def main(ideallanguage, region_descriptions, region_graphs, increase_corpus_flag = False):

    # These are the mapping ids from the ideallangueg/visualGenome to the new ids
    ids = [(1, 1), (3, 2), (4, 3), (71, 4), (9, 5),
           (2410753, 6), (713137, 7), (2412620, 8), (2412211, 9),
           (186, 10), (2396154, 11), (2317468, 12)]

    logical_merged_text = []
    # logical_surface_text = []
    surface_logical_mapping = []

    matches = extract_matches(region_graphs)

    all_entities_map = {}
    for vg_id, store_id in ids:
        print(f"Processing vg_id {vg_id} store_id: {store_id}. \nLogical")

        # LOGICAL - Extract logical forms and create training data
        logical_texts = extract_logical_forms(ideallanguage, vg_id, new_situation_id=store_id, limited=True)
        logical_merged_text.extend(logical_texts)

        # In order to make it faster, I employ the matching identities to build up the surface dictionary. But also this method could be employed
        # # SURFACE - Extract surface forms for training data
        # print(f"Surface")
        # surface_texts = extract_surface_language(region_descriptions, vg_id, store_id)
        # logical_surface_text.extend(surface_texts)  # Append to the list

        # MAPPING - Extract logical representations and mappings
        print("Match and Surface")
        entity_properties_map, situation_content, entities_numerical_ids, entities_map = extract_logical_forms(ideallanguage, vg_id, new_situation_id=None, limited=True)
        # Extract the mapping between the entities_numerical_ids and surface forms
        region_graph_mapping = extract_graph_mapping(entities_numerical_ids, matches)
        # Match the surface and logical forms
        mapping = match_logical_surface_forms(region_graph_mapping, entity_properties_map)
        surface_logical_mapping.append(mapping)
        all_entities_map[store_id] = entities_map

    if increase_corpus_flag is True:
        aug_logical_merged_text = []
        aug_surface_logical_mapping = []
        store_id = 13
        new_situation_ids, test_situation_id, training_situation_id = increase_the_corpus(ideallanguage, all_entities_map)
        print('NEW_SITUATION_IDS',new_situation_ids)
        for referent_situation_id, new_situation_id in new_situation_ids.items():
            print(f"Processing new_situation_id {new_situation_id}, increasing referent_situation_id {referent_situation_id}, with store_id: {store_id}. \nLogical")

            aug_logical_texts = extract_logical_forms(file_path= ideallanguage, situation_id=new_situation_id, new_situation_id=store_id, limited=True)
            aug_logical_merged_text.extend(aug_logical_texts)

            aug_entity_properties_map, aug_situation_content, aug_entities_numerical_ids, aug_entities_map = extract_logical_forms(ideallanguage, new_situation_id, new_situation_id=None, limited=True)
            aug_region_graph_mapping = extract_graph_mapping(aug_entities_numerical_ids, matches)
            aug_mapping = match_logical_surface_forms(aug_region_graph_mapping, aug_entity_properties_map)
            aug_surface_logical_mapping.append(aug_mapping)

            store_id += 1

    ################ Writing to files

    if increase_corpus_flag is True:
        logic_file_names = {'./data/training/extracted_logical.txt': logical_merged_text, './data/training/extracted_logic_aug.txt':aug_logical_merged_text}
        surface_file_names = {'./data/training/extracted_surface.txt': surface_logical_mapping, './data/training/extracted_surface_aug.txt':aug_surface_logical_mapping}
        logic_to_surface_file_names = {'./data/training/extracted_logical_to_surface.txt': surface_logical_mapping, './data/training/extracted_logical_to_surface_aug.txt':aug_surface_logical_mapping}
        surface_to_logic_file_names = {'./data/training/extracted_surface_to_logical.txt': surface_logical_mapping, './data/training/extracted_surface_to_logical_aug.txt':aug_surface_logical_mapping}
        sandwich_names = {'./data/training/extracted_sandwich.txt': surface_logical_mapping, './data/training/extracted_sandwich_aug.txt':aug_surface_logical_mapping}
    else:
        logic_file_names = {'./data/training/extracted_logical.txt': logical_merged_text}
        surface_file_names = {'./data/training/extracted_surface.txt': surface_logical_mapping}
        logic_to_surface_file_names = {'./data/training/extracted_logical_to_surface.txt': surface_logical_mapping}
        surface_to_logic_file_names = {'./data/training/extracted_surface_to_logical.txt': surface_logical_mapping}
        sandwich_names = {'./data/training/extracted_sandwich.txt': surface_logical_mapping}

    for logic_file_name in logic_file_names.keys():
        with open(logic_file_name, "w", encoding="utf-8") as file:
            mapping = logic_file_names[logic_file_name]
            file.write(''.join(mapping))

    # Write surface merged text to file

    for surface_file_name in surface_file_names.keys():
        with open(surface_file_name, "w", encoding="utf-8") as file:
            # file.write("".join(logical_surface_text))  # Could also be used, if we use the other method
            mapping = surface_file_names[surface_file_name]
            for situation in mapping:
                situation_items = list(situation.items())
                total_situation_bot_text = []
                for hum_text, bot_text in situation_items:
                    if isinstance(bot_text, (set, list)):
                        total_situation_bot_text.extend(bot_text)
                for i in range(len(total_situation_bot_text)):
                    bot_text = total_situation_bot_text[i]
                    file.write(f'<script.{(mapping.index(situation))+1} type=CONV>\n')
                    file.write(f'<u speaker=HUM>{bot_text}</u>\n')
                    # I do the increasing in this way such to have a mapping AABB BBCC etc.
                    if i + 1 < len(total_situation_bot_text):
                        new_bot_text = total_situation_bot_text[i + 1]
                        file.write(f'<u speaker=BOT>{new_bot_text}</u>\n')
                    else: # If there is no pair, go back to the first utterance
                        old_bot_text = total_situation_bot_text[0]
                        file.write(f'<u speaker=BOT>{old_bot_text}</u>\n')
                    file.write(f'</script.{(mapping.index(situation))+1}>\n\n')

    for logic_to_surface_file_name in logic_to_surface_file_names.keys():
        with open(logic_to_surface_file_name, "w", encoding="utf-8") as file:
            # file.write("".join(logical_surface_text))  # Could also be used, if we use the other method
            mapping = logic_to_surface_file_names[logic_to_surface_file_name]
            for situation in mapping:
                for hum_text, bot_text in situation.items():
                    for i in bot_text:
                        file.write(f'<a script.{(mapping.index(situation))+1} type=DSC>\n')
                        file.write(f'<u speaker=HUM>{hum_text}</u>\n')
                        file.write(f'<u speaker=BOT>{i}</u>\n')
                        file.write(f'</a>\n\n')

    for logic_to_surface_file_name in logic_to_surface_file_names.keys():
        with open(logic_to_surface_file_name, "w", encoding="utf-8") as file:
            mapping = surface_to_logic_file_names[logic_to_surface_file_name]
            for situation in mapping:
                for hum_text, bot_text in situation.items():
                    for i in bot_text:
                        file.write(f'<a script.{(mapping.index(situation))+1} type=DSC>\n')
                        file.write(f'<u speaker=HUM>{i}</u>\n')
                        file.write(f'<u speaker=BOT>{hum_text}</u>\n')
                        file.write(f'</a>\n\n')

    for sandwich_name in sandwich_names:
        with open(sandwich_name, "w", encoding="utf-8") as file:
            mapping = sandwich_names[sandwich_name]
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
                        file.write(f'</a>\n\n')
                    else:
                        old_logical, old_surface = flattened_items[0]
                        file.write(f"<u speaker=BOT>{old_surface}</u>\n")
                        file.write(f"<u speaker=BOT>{old_logical}</u>\n")
                        file.write(f'</a>\n\n')

if __name__ == "__main__":
    main("./data/ideallanguage.txt", "../../vgnlp2/obs/region_descriptions.json.obs", "../../vgnlp2/dsc/region_graphs.json.dsc", increase_corpus_flag = True)