'''
This script provides core utilities for transforming Visual Genome (VG) `ideallanguage` and `region_graph` 
data into multiple conversational formats for training dialogue systems like Botchen.

It includes tools for extracting logical forms, converting between logic and surface language, 
generating conversational scripts, and augmenting the corpus by finding similar situations 
based on entity overlap.

Core Capabilities:
------------------
1. **Logic → Logic**: Extract structured entities and their properties from VG logical scenes.
2. **Surface → Surface**: Convert region descriptions into conversational surface-form utterances.
3. **Logic ↔ Surface**: Match logical entities with surface descriptions using region graphs.
4. **Sandwich Format**: Mixed sequences with logic and surface alternation.
5. **Corpus Augmentation**: Expand training/test sets by mining semantically similar VG scenes.

Main Functions:
---------------
- `extract_logic_language`: Parses `ideallanguage.txt` for logic-based entity-property structures.
- `extract_surface_language`: Converts region surface descriptions into dialogues.
- `extract_surface_logic_utterances`: Extracts logic-surface form pairs from region graphs.

- `filter_region_graph_mapping`: Filters surface-form mappings by known VG entity IDs.
- `match_logical_surface_forms`: Links logic and surface forms into reusable mappings.

- `write_logic_to_surface`: Writes logic→surface (or inverse) training files.
- `write_surface`: Writes conversational surface-surface files.
- `write_sandwich`: Creates "sandwich" format dialogues with alternating logic/surface content.
- `increase_the_corpus`: Expands dataset with similar VG scenes using entity overlap metrics.

Key Parameters and Flags:
--------------------------
- `limited`, `limited_max_utterances`: Restrict number of extracted utterances (test mode).
- `new_situation_id`: ID to assign to new dialogue block.
- `reverse`: Flag to swap HUM/BOT roles for logic-surface tasks.
- `write_all_files`: If True, writes all outputs to disk; otherwise returns string.
- `min_referent_overlap_ratio`, `min_target_overlap_ratio`: Thresholds for corpus augmentation similarity.
- `train_split_ratio`: Proportion of augmented data used for training (vs. test).
- `max_per_referent`: Max similar scenes to retrieve per original situation.

***

Functions description:

(A) extract_logic_language(file_path, situation_id, new_situation_id=None, limited=False, limited_max_utterances=None)
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

import re
import random
import os
import argparse
from collections import defaultdict
import logging
logging.basicConfig(level=logging.INFO)

''' 
LOGIC TO LOGIC
Function to extract a specific situation from the Visual Genome dataset and transform it into a conversational format.
Also, to extract entities and properties
Ideallanguage is the input file: './data/ideallanguage.txt'
'''

# file_path:str, situation_id:str, new_situation_id:Optional[str]=None, limited:bool=False
def extract_logic_language(file_path, situation_id, new_situation_id, limited, limited_max_utterances):
    # logging.info('Logic-logic mapping process began')
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
        # logging.info('Logic-logic mapping process finished [script_output]')
        return script_output

    # DEFAULT: Return a dictionary of entity names and their properties
    if new_situation_id is None:
        entity_summary_map = {}
        for entity_id, name in entity_id_to_name.items():
            entity_summary_map[entity_id] = f"{name} {' '.join(entity_id_to_properties[entity_id])}"
        # logging.info('Logic-logic mapping process finished [mapping only]')
        return entity_summary_map, situation_body, entity_ids_in_block, entity_id_to_name
# entity_summary_map: Dict[str, str], situation_body: str, entity_ids_in_block: List[str], entity_id_to_name: Dict[str, str]
# script_output: str

'''
SURFACE TO SURFACE
Extract the surface languages from the utterances for the situation
Region descriptions is the input file: './obs/region_descriptions.json.obs'
This function is not really used but it could be useful.
'''

# file_path:str, idx_to_extract:str, output_idx:str
def extract_surface_language(file_path, idx_to_extract, output_idx):
    # logging.info('Surface-surface mapping process began')
    with open(file_path, 'r') as file:
        situation_pattern = rf"<a type=OBS idx={idx_to_extract}>(.*?)</a>"
        situation_data = re.search(situation_pattern, file.read(), re.DOTALL)
    if not situation_data:
        # logging.info(f"No situation found with id={idx_to_extract}")
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
    # logging.info('Surface-surface mapping process finished')
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
    # logging.info('Surface-logic mapping process began')
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
    # logging.info('Surface-logic mapping process finished')
    return new_map
# new_map: dict[str, set[str]]

'''
CREATE TRAINING FILES
'''

def write_logic_to_surface(file_path, mapping, plus_index, reverse=False, write_all_files=False):
    lines = []
    for situation in mapping:
        for hum_text, bot_texts in situation.items():
            for bot_text in bot_texts:
                lines.append(f'<a script.{mapping.index(situation) + plus_index} type=DSC>')
                if reverse:
                    lines.append(f'<u speaker=HUM>{bot_text}</u>')
                    lines.append(f'<u speaker=BOT>{hum_text}</u>')
                else:
                    lines.append(f'<u speaker=HUM>{hum_text}</u>')
                    lines.append(f'<u speaker=BOT>{bot_text}</u>')
                lines.append('</a>\n')
    content = '\n'.join(lines)
    if write_all_files is True:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
        return content
    else:
        return content

def write_surface(file_path, mapping, plus_index, write_all_files=False):
    lines = []
    for situation in mapping:
        situation_items = list(situation.items())
        total_bot_texts = []
        for hum_text, bot_text in situation_items:
            if isinstance(bot_text, (set, list)):
                total_bot_texts.extend(bot_text)
        for i in range(len(total_bot_texts)):
            bot_text = total_bot_texts[i]
            lines.append(f'<script.{mapping.index(situation) + plus_index} type=CONV>')
            lines.append(f'<u speaker=HUM>{bot_text}</u>')
            if i + 1 < len(total_bot_texts):
                new_bot_text = total_bot_texts[i + 1]
                lines.append(f'<u speaker=BOT>{new_bot_text}</u>')
            else:
                old_bot_text = total_bot_texts[0]  # fallback
                lines.append(f'<u speaker=BOT>{old_bot_text}</u>')
            lines.append(f'</script.{mapping.index(situation) + plus_index}>\n')
    content = '\n'.join(lines)
    if write_all_files is True:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
        return content
    else:
        return content

def write_sandwich(file_path, mapping, plus_index, write_all_files=False):
    lines = []
    for situation_idx, situation in enumerate(mapping):
        flattened_items = []
        for logical_form, surface_form_set in situation.items():
            for surface_form in surface_form_set:
                flattened_items.append([logical_form, surface_form])
        for i in range(len(flattened_items)):
            lines.append(f'<a script.{situation_idx + plus_index} type=SDW>')
            logical_form, surface_form = flattened_items[i]
            lines.append(f"<u speaker=HUM>{logical_form}</u>")
            lines.append(f"<u speaker=BOT>{surface_form}</u>")
            if i + 1 < len(flattened_items):
                next_logical, next_surface = flattened_items[i + 1]
                lines.append(f"<u speaker=BOT>{next_surface}</u>")
                lines.append(f"<u speaker=BOT>{next_logical}</u>")
            else:
                old_logical, old_surface = flattened_items[0]
                lines.append(f"<u speaker=BOT>{old_surface}</u>")
                lines.append(f"<u speaker=BOT>{old_logical}</u>")
            lines.append(f'</a>\n')
    content = '\n'.join(lines)     
    if write_all_files is True:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
        return content
    else:
        return content

'''
INCREASE DATA AMOUNT
'''

def increase_the_corpus(
        ideallanguage_path,
        original_situation_ids,
        all_entities_map,
        min_referent_overlap_ratio,
        min_target_overlap_ratio,
        min_content_length,
        max_content_length,
        max_per_referent,
        train_split_ratio
):
    
    # Here I am substituting the list got from all_entities_map with the original tuples names
    key_mapping = {a: b for b, a in original_situation_ids}
    adjusted_entities_map = {key_mapping[k]: v for k, v in all_entities_map.items()}
    original_ids_list = [k for k,v in original_situation_ids]

    with open(ideallanguage_path, 'r') as file:
        content = file.read()

    situation_pattern = r"<situation id=(.*?)>(.*?)</situation>"
    all_situations = re.findall(situation_pattern, content, re.DOTALL)

    all_situation_id_dict = defaultdict(list)
    training_situation_id_dict = defaultdict(list)
    test_situation_id_dict = defaultdict(list)

    # Look at the situations from which we are augmenting
    for referent_id, (original_id, referent_entity_map) in enumerate(adjusted_entities_map.items(), start=1):

        referent_entities = list(referent_entity_map.values())
        referent_set = set(referent_entities)
        # logging.info(f'Training Referent Situation {referent_id}, Original VG Situation {original_id} - Reference Entities: {referent_set}')
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
                if (overlap_ratio_ref >= min_referent_overlap_ratio and
                    overlap_ratio_tgt >= min_target_overlap_ratio and
                    min_content_length <= len(situation_content) <= max_content_length):
                    matched_ids.append(target_id)

        for id in original_ids_list:
            original_id_str = str(id)
            if original_id_str in matched_ids:
                matched_ids.remove(original_id_str)

        # logging.info(f"Training Referent Situation {referent_id}, Original VG Situation {original_id} matched with {len(matched_ids)} other vg situations: {matched_ids}")
        # If there is the possibility, divide between training and testing items. In any case, build a dictionary with all the items
        if matched_ids:
            selected = random.sample(matched_ids, min(len(matched_ids), max_per_referent))
            split_idx = int(len(selected) * (1 - train_split_ratio))
            test_situation_id_dict[referent_id] = selected[:split_idx]
            training_situation_id_dict[referent_id] = selected[split_idx:]
            all_situation_id_dict[referent_id] = selected
            # logging.info(f"Selected {len(selected)}")
        else:
            logging.info("We couldn't find enough matches, auch")

    # Plain lists with ids to retrieve from the ideallanguage for incrementing the corpus, with no store_id attached
    all_augmenting_situation_ids = []
    training_augmenting_situation_ids = []
    testing_augmenting_situation_ids = []
    for augmenting_situation_ids in all_situation_id_dict.values():
        all_augmenting_situation_ids.extend(augmenting_situation_ids)
    for training_situation_ids in training_situation_id_dict.values():
        training_augmenting_situation_ids.extend(training_situation_ids)
    for testing_situation_ids in test_situation_id_dict.values():
        testing_augmenting_situation_ids.extend(testing_situation_ids)

    # Tuple lists with ids to retrieve from the ideallanguage for incrementing the corpus and their store_id attached
    final_augmenting_situation_ids = []
    final_training_situation_ids = []
    final_testing_situation_ids = []
    base_index = len(all_situation_id_dict) + 1

    for offset, item in enumerate(all_augmenting_situation_ids):
        final_augmenting_situation_ids.append((int(item), base_index + offset)) # It should be the len of the original situations chosen (since it's s dict mapping from that)
    for offset, item in enumerate(training_augmenting_situation_ids):
        final_training_situation_ids.append((int(item), base_index + offset)) # I am counting from here cause I will add them to the original situations for the training and I do not want numbers to mess up
    for offset, item in enumerate(testing_augmenting_situation_ids):
        new_base_index = len(training_augmenting_situation_ids) + base_index
        final_testing_situation_ids.append((int(item), new_base_index+offset))

    logging.info(f'We have found {len(all_augmenting_situation_ids)} total situations, '
                 f'{len(training_augmenting_situation_ids)} training situations '
                 f'and {len(testing_augmenting_situation_ids)} testing situation we can use to increase the corpus')

    return final_augmenting_situation_ids, final_training_situation_ids, final_testing_situation_ids