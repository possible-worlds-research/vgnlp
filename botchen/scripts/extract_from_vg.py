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

# Function to extract a specific situation from the Visual Genome dataset and transform it into a conversational format.
# Also, to extract entities and properties
# Ideallanguage is the input file: './data/ideallanguage.txt'
def extract_logical_forms(file_path, situation_id, new_situation_id=None):
    with open(file_path, 'r') as file:
        content = file.read()

    # Extract the situation block by matching its ID using a regular expression pattern.
    situation_pattern = rf"<situation id={situation_id}>(.*?)</situation>"
    match = re.search(situation_pattern, content, re.DOTALL)

    if match:

        situation_content = match.group(1)

        # Dictionaries to store the entity details: entity names and their properties.
        entity_map = {}  # Maps entity ID -> entity name
        properties_map = {}  # Maps entity ID -> list of properties

        # Store the numerical id of the entities
        entity_id_pattern = r'<entity id=(\d+)>'
        entity_matches = re.findall(entity_id_pattern, situation_content)

        # First pass: Extract and store each entity's ID, name, and initialize its properties.
        entity_pattern = re.compile(r"<entity id=(\d+)>\s*(.*?)\s*</entity>", re.DOTALL)
        for entity_match in entity_pattern.finditer(situation_content):
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

        # Second pass: Extract properties and relations between entities.
        for entity_match in entity_pattern.finditer(situation_content):
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

        if new_situation_id:
            # Construct the conversation script using the extracted data.
            script_content = f"<script.{new_situation_id} type=CONV>\n"
            current_speaker = "HUM"
            for entity_id, entity_name in entity_map.items():
                properties_str = " ".join(re.sub(r"\.n\.\d+", "", prop) for prop in properties_map[entity_id])
                script_content += f'<u speaker={current_speaker}>({entity_name} {properties_str})</u>\n'

                # Alternate between human (HUM) and bot (BOT) speakers.
                current_speaker = "BOT" if current_speaker == "HUM" else "HUM"

            script_content += f"</script.{new_situation_id}>\n\n"
            return script_content

        # Extract the object id numbers, matching the situations so that the situation id is matched with the entities id
        if new_situation_id is None:
            # Extract a mapping object with the ids
            unified_map = {}
            for entity_id, entity_name in entity_map.items():
                # Combine the entity name and properties into a single string
                properties_str = " ".join(properties_map[entity_id])
                unified_map[entity_id] = f"{entity_map[entity_id]} {properties_str}"

            return unified_map, situation_content, entity_matches

    if not match:
        print(f"No situation found with id={situation_id}")
        return None

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
    new_file = f"<script.{output_idx} type=CONV>\n"
    current_speaker = 'HUM'
    for statement in statements:
        new_file += f"<u speaker={current_speaker}>{statement.strip()}</u>\n"
        current_speaker = 'BOT' if current_speaker == 'HUM' else 'HUM'
    new_file += f"</script.{output_idx}>\n\n"
    return new_file

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
        new_map[key] = description
    return new_map

# Main function to execute the extraction process for multiple situations and save the result.
def main(ideallanguage, region_descriptions, region_graphs):

    ids = [(1, 1), (3, 2), (4, 3)]

    # ids = [(1, 1), (3, 2), (4, 3), (71, 4), (9, 5),
    #        (2410753, 6), (713137, 7), (2412620, 8), (2412211, 9),
    #        (186, 10), (2396154, 11), (2317468, 12)]

    logical_merged_text = []  # Should be a list to store text pieces
    logical_surface_text = []  # Same for surface text

    surface_logical_mapping = []  # List to store mappings
    matches = extract_matches(region_graphs)  # Extract mappings

    for vg_id, store_id in ids:
        print(f"Processing vg_id {vg_id} store_id: {store_id}. \n Logical")

        # LOGICAL - Extract logical forms and create training data
        logical_texts = extract_logical_forms(ideallanguage, vg_id, new_situation_id=store_id)
        logical_merged_text.extend(logical_texts)  # Append to the list

        # SURFACE - Extract surface forms for training data
        print(f"Surface")
        surface_texts = extract_surface_language(region_descriptions, vg_id, store_id)
        logical_surface_text.extend(surface_texts)  # Append to the list

        print("Match")
        # MAPPING - Extract logical representations and mappings
        entity_properties_map, situation_content, entities_numerical_ids = extract_logical_forms(ideallanguage, vg_id, new_situation_id=None)

        # Extract the mapping between the entities_numerical_ids and surface forms
        region_graph_mapping = extract_graph_mapping(entities_numerical_ids, matches)

        # Match the surface and logical forms
        mapping = match_logical_surface_forms(region_graph_mapping, entity_properties_map)  # Ensure correct reference here
        surface_logical_mapping.append(mapping)  # Append the mapping

    # Write logical merged text to file
    with open('./data/training/extracted_scripts_logical.txt', "w", encoding="utf-8") as file:
        file.write(''.join(logical_merged_text))  # Join the list items into a single string with line breaks

    # Write surface merged text to file
    with open('./data/training/extracted_scripts_surface.txt', "w", encoding="utf-8") as file:
        file.write("".join(logical_surface_text))  # Same here, join list into string

    # Write the mappings to the dsc file
    with open('./data/training/extracted_scripts_dsc.txt', "w", encoding="utf-8") as file:
        for situation in surface_logical_mapping:
            # Loop through each item in the mapping (assuming it's a dictionary)
            file.write(f'<a script.{(surface_logical_mapping.index(situation))+1} type=DSC>\n')
            for hum_text, bot_text in situation.items():
                # Write the formatted text for each entry
                file.write(f'<u speaker=HUM>{hum_text}</u>\n')
                file.write(f'<u speaker=BOT>{bot_text}</u>\n')
            file.write(f'</a script.{(surface_logical_mapping.index(situation))+1}>\n\n')

# Ensure the script runs when executed directly.
if __name__ == "__main__":
    logical_representation, situation_content, items = extract_logical_forms("./data/ideallanguage.txt", '1', new_situation_id=None)
    # print(situation_content)
    main("./data/ideallanguage.txt", "../../vgnlp2/obs/region_descriptions.json.obs", "../../vgnlp2/dsc/region_graphs.json.dsc")


