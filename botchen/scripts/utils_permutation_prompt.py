'''
This script includes functions for creating permutations.

Main Functional Areas:
-----------------------
1. **General Utilities**:
   - `substitute_word`, `apply_substitutions`: Perform word-level substitutions in dialogue texts.
   - `extract_situations`: Breaks a text corpus into indexed dialogue blocks.
   - `extract_entities_properties`: Extracts entity-property structures.

2. **Substitution and Synonym Expansion**:
   - `generate_random_substitutions`: Randomly selects alternatives from a synonym list.
   - `get_conceptnet_hypernyms_synonyms`: Uses ConceptNet API to retrieve synonyms and hypernyms
     (supports automated data augmentation).

3. **Logic-to-Logic and Surface-to-Surface Conversion**:
   - `prompt_surface_logic`: Extracts unique utterances and formats them as prompt data. 
   - `permutation_surface_logic`: Creates augmented data by substituting terms within surface-form conversations.

4. **Logic--Surface and Sandwich Format Conversion**:
   - `permutation_sandwich_logic_surface_transl`: Applies substitutions to sandwich, Logic--Surface data.
   - `prompt_sandwich_logic_surface_transl`: Extracts utterances from sandwich or logic--surface format and 
     reformats them for prompting.

External Dependencies:
------------------------
- Requires network access to query ConceptNet API: `http://api.conceptnet.io`
'''

import re
import random
import os
import argparse
from collections import defaultdict
import requests
import random
import logging

################# General functions 

def substitute_word(script_text, old_word, new_word):
    result = re.sub(rf'\b{re.escape(old_word)}\b', new_word, script_text)
    return result if result != script_text else None

def apply_substitutions(texts, substitutions):
    if isinstance(texts, str):
        texts = [texts]
    new_texts = texts.copy()
    for i, text in enumerate(texts):
        for old, new in substitutions:
            if old:
                new_texts[i] = re.sub(rf'\b{re.escape(old)}\b', new, new_texts[i])

    if new_texts == texts:
        return (None,) * len(texts)
    return tuple(new_texts)

def _format_tag(script_num, hum_text, bot_texts, sandwich_flag=None):
    tag_type = "SDW" if sandwich_flag else "DSC"
    output = f"<a script.{script_num} type={tag_type}>\n"
    output += f"<u speaker=HUM>{hum_text}</u>\n"
    for bot_text in bot_texts:
        output += f"<u speaker=BOT>{bot_text}</u>\n"
    output += "</a>\n\n"
    return output

def extract_situations(content):
    scripts = re.findall(r"(<script\.\d+ type=CONV>.*?</script\.\d+>)", content, re.DOTALL)
    situations = {}
    script_counts = defaultdict(int)
    for script_content in scripts:
        match = re.search(r"<script\.(\d+) type=CONV>.*?</script\.\d+>", script_content, re.DOTALL)
        if match:
            script_number = int(match.group(1))
            script_counts[script_number] += 1
            sub_number = script_counts[script_number]
            key = f"situation {script_number}.{sub_number}"
            situations[key] = script_content
    return situations

def extract_entities_properties(content, surface_language=False):
    if surface_language:
        utterances = re.findall(r'<u speaker=[^>]*>(.*?)</u>', content)
        return list(set(utterances))

    entity_properties = {}
    matches = re.findall(r'<u speaker=[^>]*>\((.*?)\)</u>', content)

    for utterance in matches:
        parts = utterance.split()
        if not parts: continue
        entity, *properties = parts
        clean_props = {re.sub(r'\.\d+', '', prop) for prop in properties}
        entity_properties.setdefault(entity, set()).update(clean_props)

    return entity_properties

################# Substitution dictionary

def generate_random_substitutions(substitutions_dict):
    if not isinstance(substitutions_dict, dict):
        raise ValueError("Expected a dictionary of substitutions")
    return {
        key: random.choice([v for v in values if v != key]) if any(v != key for v in values) else key
        for key, values in substitutions_dict.items()
    }
    
def get_conceptnet_hypernyms_synonyms(term_list):
    term_list_results = {}
    for term in term_list:
        url = f"https://api.conceptnet.io/c/en/{term}?offset=0&limit=1000"
        try:
            obj = requests.get(url).json()
        except Exception as e:
            logging.info(f"Error fetching from ConceptNet: {e}")
            return []
        term_results = set()
        # Iterate through all semantic edges (relations) associated with the term
        for edge in obj['edges']:
            # Edge direction originating from the term
            if edge['start']['term'] != f'/c/en/{term}':
                continue
            if not edge['end']['term'].startswith('/c/en/'):
                continue
            rel_label = edge['rel']['label']
            target_label = edge['end']['label'].lower()
            if len(target_label.split()) > 1:
                continue
            if rel_label in {'Synonym', 'IsA'}:
                term_results.add(target_label)
        term_list_results[term] = list(term_results)

    substitutions_dict = generate_random_substitutions(term_list_results)

    final_list = list(substitutions_dict.items())

    return final_list

################# Logic to logic and Surface to surface

# Function to generate a prompt script from the extracted entities and their properties
def prompt_surface_logic(content):

    scripts = re.findall(r'<script\.(\d+) type=CONV>(.*?)</script\.\1>', content, re.DOTALL)
    merged_scripts = defaultdict(set)
    for script_number, script_text in scripts:
        utterances = re.findall(r'<u speaker=[^>]+>(.*?)</u>', script_text)
        merged_scripts[int(script_number)].update(utterances)

    prompt_text = ''
    for script_number, script_body in merged_scripts.items():
        prompt_text += f'<script.{script_number} type=CONV>\n'
        for utterance in script_body:
            prompt_text += f'<u speaker=HUM>{utterance}</u>\n'
        prompt_text += f'</script.{script_number}>\n\n'

    return prompt_text

def permutation_surface_logic(situations, substitutions):

    new_situations = {}
    prompt_content_total = ''
    merged_text_total = '' 

    # Process each situation and apply substitutions
    for script_id, script_text in situations.items():
        match = re.search(r'(\d+)\.(\d+)', script_id)
        if match:
            script_number = int(match.group(1))
            script_index = int(match.group(2))
        for index, (old_word, new_word) in enumerate(substitutions, start=1):
            modified_text = substitute_word(script_text, old_word, new_word)
            if modified_text: 

                version_name = f"situation {script_number}.{script_index}.{index}"
                new_situations[version_name] = modified_text

    situations.update(new_situations)
    situations = dict(sorted(situations.items(), key=lambda x: list(map(int, re.findall(r'\d+', x[0])))))

    for content in situations.values():
        merged_text_total += content + '\n\n'

    return merged_text_total

################## Logic to surface and sandwich

def permutation_sandwich_logic_surface_transl(input_text, substitutions, sandwich_flag=None):

    pattern = (
        r'<a script\.(\d+) type=SDW>\s*<u speaker=HUM>(.*?)</u>'
        r'\s*<u speaker=BOT>(.*?)</u>\s*<u speaker=BOT>(.*?)</u>'
        r'\s*<u speaker=BOT>(.*?)</u>\s*</a>'
        if sandwich_flag else
        r'<a script\.(\d+) type=DSC>\s*<u speaker=HUM>(.*?)</u>'
        r'\s*<u speaker=BOT>(.*?)</u>\s*</a>'
    )

    matches = re.findall(pattern, input_text, re.DOTALL)
    processed_text = ""

    for match in matches:
        script_num = int(match[0])  # Get the script number

        hum_text = match[1]  # Get the human text
        bot_texts = list(match[2:])

        texts_to_process = [hum_text] + bot_texts

        result = apply_substitutions(texts_to_process, substitutions)
        processed_text += _format_tag(script_num, hum_text, bot_texts, sandwich_flag)

        if result and any(x is not None for x in result):
            new_hum = result[0] if result[0] is not None else hum_text
            new_bots = [
                result[i] if result[i] is not None else bot_texts[i - 1]
                for i in range(1, len(result))
            ]
            processed_text += _format_tag(script_num, new_hum, new_bots, sandwich_flag)

    return processed_text

# Function to extract unique HUM utterances per script number
def prompt_sandwich_logic_surface_transl(input_text, sandwich_flag=None):

    tag_type = 'SDW' if sandwich_flag else 'DSC'
    pattern = rf'<a script\.(\d+) type={tag_type}>(.*?)</a>'

    hum_pattern = r'<u speaker=HUM>(.*?)</u>'
    bot_pattern = r'<u speaker=BOT>([^()<>\n]+)</u>'
    
    script_groups = {}

    for script_num, content in re.findall(pattern, input_text, re.DOTALL):
        hum_utterances = set(re.findall(hum_pattern, content))
        
        if sandwich_flag:
            utterances = {utt.strip() for utt in re.findall(hum_pattern, content) if utt.strip()}
            hum_utterances.update(utterances)

        script_groups.setdefault(script_num, set()).update(hum_utterances)

    output_lines = []
    for script_num, utterances in script_groups.items():
        output_lines.append(f'<a script.{script_num} type={tag_type}>')
        output_lines.extend(f'<u speaker=HUM>{utt}</u>' for utt in utterances)
        output_lines.append('</a>\n')

    output_text = '\n'.join(output_lines)

    return output_text





###############
################ OTHER UTILS

def make_csv_from_data(file_path, name):
    with open(file_path, 'r') as file:
        data = file.read()
    evaluation_pattern = re.compile(r"Evaluating with framework: (\d+), script eval ID: (\d+), script optimal ID: (\d+)")
    row_similarity_pattern = re.compile(r"Average Rows Similarity:\s+([\d\.]+)")
    column_similarity_pattern = re.compile(r"Average Column Similarity:\s+([\d\.]+)")

    summary_data = []
    evaluations = data.strip().split("NEWNEW")

    for eval_block in evaluations[0:]:  # Skip empty first split
        eval_info = evaluation_pattern.search(eval_block)
        row_sim = row_similarity_pattern.search(eval_block)
        col_sim = column_similarity_pattern.search(eval_block)

        if eval_info and row_sim and col_sim:
            framework, eval_script, opt_script = eval_info.groups()
            avg_row_sim = float(row_sim.group(1))
            avg_col_sim = float(col_sim.group(1))

            summary_data.append([framework, eval_script, opt_script, avg_row_sim, avg_col_sim])

    summary_df = pd.DataFrame(summary_data, columns=["Framework", "Eval Script", "Optimal Script", "Avg Row Similarity", "Avg Column Similarity"])
    summary_df.to_csv(f"{name}", index=False)

def heatmap(file_path, row_column, eval_optimal, number_eval, log_file):
    # row_column= Row/Column based on what you want to evaluate
    # eval_optimal= Optimal/Eval based on what you are comparing
    # number_eval = 1/2 if evaluating with 1 or 2 eval
    df = pd.read_csv(file_path)
    heatmap_data = df.pivot_table(index='Eval Script', columns='Optimal Script', values=f'Avg {row_column} Similarity')
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="Blues", linewidths=0.5)
    plt.title(f'EVAL {number_eval} - Heatmap of Avg {row_column} Similarity')
    plt.xlabel('Optimal Script')
    plt.ylabel(f'{eval_optimal} Script')
    plt.savefig(log_file)

# Previous substitution item, just to have an idea 

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