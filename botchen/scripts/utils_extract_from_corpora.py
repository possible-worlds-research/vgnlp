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

def read_situation_block(file_path, situation_id):
    with open(file_path, 'r') as file:
        content = file.read()
    match = re.search(rf"<situation id={situation_id}>(.*?)</situation>", content, re.DOTALL)
    return match.group(1) if match else None

def extract_entity_blocks(situation_body, limited=False, limit=0):
    entity_blocks = re.findall(r"<entity id=(\d+)>\s*(.*?)\s*</entity>", situation_body, re.DOTALL)
    return entity_blocks[:limit] if limited else entity_blocks

def extract_entity_names(entity_blocks):
    id_to_name = {}
    for entity_id, block in entity_blocks:
        lines = block.strip().split("\n")
        match = re.match(r"([\w\.]+)\(\d+\)", lines[0].strip())
        if match:
            raw_name = match.group(1)
            norm_name = re.sub(r"\.n\.\d+", ".n", raw_name)
            id_to_name[entity_id] = norm_name
    return id_to_name

def extract_entity_properties(entity_blocks, id_to_name):
    id_to_properties = {eid: [] for eid in id_to_name}
    for entity_id, block in entity_blocks:
        lines = block.strip().split("\n")[1:]  # Skip name line
        for line in lines:
            match = re.match(r"([\w|]+)\(([\d,]+)\)", line.strip())
            if not match:
                continue
            rel_or_prop, ids_str = match.groups()
            ref_ids = ids_str.split(",")

            if len(ref_ids) > 1:
                source_id, target_id = ref_ids
                if source_id == entity_id and target_id in id_to_name:
                    target_name = re.sub(r"\.n", "", id_to_name[target_id])
                    id_to_properties[entity_id].append(f"{rel_or_prop}-{target_name}")
                elif target_id == entity_id and source_id in id_to_name:
                    source_name = re.sub(r"\.n", "", id_to_name[source_id])
                    id_to_properties[entity_id].append(f"{source_name}-{rel_or_prop}")
            else:
                clean_name = re.sub(r"\.n", "", rel_or_prop)
                id_to_properties[entity_id].append(clean_name)
    return id_to_properties

def generate_script_output(new_situation_id, id_to_name, id_to_properties):
    script_output = ''
    entities = list(id_to_name.items())

    for i, (eid, name) in enumerate(entities):
        props = id_to_properties.get(eid, [])
        clean_props = [re.sub(r"\.n\.\d+", "", p) for p in props]
        unique_props = list(dict.fromkeys(clean_props))

        script = f"<script.{new_situation_id} type=CONV>\n"
        script += f'<u speaker=HUM>({name} {" ".join(unique_props)})</u>\n'

        if i + 1 < len(entities):
            next_eid, next_name = entities[i + 1]
        else:
            next_eid, next_name = entities[0]

        next_props = id_to_properties.get(next_eid, [])
        next_clean_props = [re.sub(r"\.n\.\d+", "", p) for p in next_props]
        next_unique_props = list(dict.fromkeys(next_clean_props))

        script += f'<u speaker=BOT>({next_name} {" ".join(next_unique_props)})</u>\n'
        script += f"</script.{new_situation_id}>\n\n"

        script_output += script

    return script_output

def extract_logic_language(file_path, situation_id, new_situation_id=None, limited=False, limited_max_utterances=0):
    situation_body = read_situation_block(file_path, situation_id)
    if not situation_body:
        logging.info(f"No situation found with id={situation_id}")
        return None

    entity_blocks = extract_entity_blocks(situation_body, limited, limited_max_utterances)
    id_to_name = extract_entity_names(entity_blocks)
    id_to_properties = extract_entity_properties(entity_blocks, id_to_name)
    entity_ids = [eid for eid, _ in entity_blocks]

    if new_situation_id:
        return generate_script_output(new_situation_id, id_to_name, id_to_properties)

    entity_summary_map = {
        eid: f"{name} {' '.join(id_to_properties[eid])}" for eid, name in id_to_name.items()
    }
    return entity_summary_map, situation_body, entity_ids, id_to_name

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

        entity_form_list = []
        for entity_id in entity_ids_list:
            if entity_id in logical_map:
                props = logical_map[entity_id].split()
                if not props:
                    continue
                entity = props[0]
                entity_props = list(dict.fromkeys(props[1:]))
                entity_str = f"{entity} {' '.join(entity_props)}".strip()
                entity_form_list.append(f"({entity_str})")

        key = ", ".join(entity_form_list)
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
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
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
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
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
            lines.append(f"<u speaker=HUM>{surface_form}</u>")
            lines.append(f"<u speaker=BOT>{logical_form}</u>")
            if i + 1 < len(flattened_items):
                next_logical, next_surface = flattened_items[i + 1]
                lines.append(f"<u speaker=BOT>{next_logical}</u>")
                lines.append(f"<u speaker=BOT>{next_surface}</u>")
            else:
                old_logical, old_surface = flattened_items[0]
                lines.append(f"<u speaker=BOT>{old_logical}</u>")
                lines.append(f"<u speaker=BOT>{old_surface}</u>")
            lines.append(f'</a>\n')
    content = '\n'.join(lines)     
    if write_all_files is True:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
        return content
    else:
        return content

'''
INCREASE DATA AMOUNT
'''

def read_all_situations(ideallanguage_path):
    with open(ideallanguage_path, 'r') as file:
        content = file.read()
    return re.findall(r"<situation id=(.*?)>(.*?)</situation>", content, re.DOTALL)

def extract_entity_names_from_situation(situation_content):
    entity_pattern = re.compile(r"<entity id=(\d+)>\s*(.*?)\s*</entity>", re.DOTALL)
    all_entities = entity_pattern.finditer(situation_content)

    entity_names = []
    for match in all_entities:
        lines = match.group(2).strip().split("\n")
        if not lines:
            continue
        name_match = re.match(r"([\w\.]+)\(\d+\)", lines[0].strip())
        if name_match:
            clean_name = re.sub(r"\.n\.\d+", ".n", name_match.group(1))
            entity_names.append(clean_name)
    return set(entity_names)

def match_situations(referent_set, all_situations, original_ids_list,
                     min_referent_overlap_ratio, min_target_overlap_ratio,
                     min_content_length, max_content_length):
    matched_ids = []

    for target_id, content in all_situations:
        target_set = extract_entity_names_from_situation(content)
        if not referent_set or not target_set:
            continue

        overlap = referent_set & target_set
        overlap_ratio_ref = len(overlap) / len(referent_set)
        overlap_ratio_tgt = len(overlap) / len(target_set)

        if (overlap_ratio_ref >= min_referent_overlap_ratio and
            overlap_ratio_tgt >= min_target_overlap_ratio and
            min_content_length <= len(content) <= max_content_length and
            target_id not in original_ids_list):
            matched_ids.append(target_id)

    return matched_ids

def split_and_sample_matched_ids(matched_ids, max_per_referent, train_split_ratio):
    selected = random.sample(matched_ids, min(len(matched_ids), max_per_referent))
    split_idx = int(len(selected) * (1 - train_split_ratio))
    return selected[:split_idx], selected[split_idx:], selected

def assign_new_ids(ids, offset_start):
    return [(int(sid), offset_start + i) for i, sid in enumerate(ids)]

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
    key_mapping = {a: b for b, a in original_situation_ids}
    adjusted_entities_map = {key_mapping[k]: v for k, v in all_entities_map.items()}
    original_ids_list = {str(k) for k, _ in original_situation_ids}

    all_situations = read_all_situations(ideallanguage_path)

    all_situation_id_dict = defaultdict(list)
    training_situation_id_dict = defaultdict(list)
    test_situation_id_dict = defaultdict(list)

    for referent_id, (original_id, referent_map) in enumerate(adjusted_entities_map.items(), start=1):
        referent_set = set(referent_map.values())

        matched_ids = match_situations(
            referent_set,
            all_situations,
            original_ids_list,
            min_referent_overlap_ratio,
            min_target_overlap_ratio,
            min_content_length,
            max_content_length
        )

        if matched_ids:
            test_ids, train_ids, all_ids = split_and_sample_matched_ids(
                matched_ids, max_per_referent, train_split_ratio
            )
            test_situation_id_dict[referent_id] = test_ids
            training_situation_id_dict[referent_id] = train_ids
            all_situation_id_dict[referent_id] = all_ids
        else:
            logging.info("We couldn't find enough matches, auch")

    # Flatten results
    all_ids_flat = [sid for sids in all_situation_id_dict.values() for sid in sids]
    train_ids_flat = [sid for sids in training_situation_id_dict.values() for sid in sids]
    test_ids_flat = [sid for sids in test_situation_id_dict.values() for sid in sids]

    # Assign new IDs
    base_index = len(all_situation_id_dict) + 1
    final_all = assign_new_ids(all_ids_flat, base_index)
    final_train = assign_new_ids(train_ids_flat, base_index)
    final_test = assign_new_ids(test_ids_flat, base_index + len(train_ids_flat))

    logging.info(f'We have found {len(all_ids_flat)} total situations, '
                 f'{len(train_ids_flat)} training situations '
                 f'and {len(test_ids_flat)} testing situations to increase the corpus')

    return final_all, final_train, final_test