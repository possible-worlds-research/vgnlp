import re
import random
import os
import argparse
from collections import defaultdict

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
    situations = {f"Situation {i+1}": script for i, script in enumerate(scripts)}
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

################# Logic to logic and Surface to surface

# Function to generate a prompt script from the extracted entities and their properties
def prompt_surface_logic(script_number, entity_properties, surface_language=False):
    lines = [f'<script.{script_number} type=CONV>']
    if surface_language:
        for utt in entity_properties:
            lines.append(f'<u speaker=HUM>{utt}</u>')
    else:
        for entity, props in entity_properties.items():
            props_str = ' '.join(props - {entity})
            lines.append(f'<u speaker=HUM>({entity}.n {props_str})</u>')
    
    lines.append(f'</script.{script_number}>\n')
    return '\n'.join(lines)

def permutation_surface_logic(situations, substitutions_per_situation, surface_language=False):

    new_situations = {}
    all_count = 0
    prompt_content_total = ''
    merged_text_total = ''

    # Process each situation and apply substitutions
    for script_id, script_text in situations.items():
        script_number = int(script_id.split()[1])
        substitutions = substitutions_per_situation.get(script_number, [])
        for index, (old_word, new_word) in enumerate(substitutions, start=1):
            modified_text = substitute_word(script_text, old_word, new_word)
            if modified_text: 
                version_name = f"Situation {script_number}.{index}"
                new_situations[version_name] = modified_text

    # Merge and save the modified situations for training purposes
    merged_text_total = ''
    prompt_content_total = ''

    script_numbers = sorted({int(k.split()[1].split('.')[0]) for k in new_situations})
    print(script_numbers)
    for script_number in script_numbers:
        merged_text = ""
        for idx in range(1, 100):  # Arbitrarily large range to collect all versions
            key = f"Situation {script_number}.{idx}"
            if key in new_situations:
                merged_text += new_situations[key] + "\n\n"
        
        if not merged_text:
            continue

        all_count += len(re.findall(r"<u speaker=", merged_text))
        merged_text_total += merged_text

        if surface_language:
            unique_utts, _ = extract_entities_properties(merged_text, surface_language=True)
            prompt_content = prompt_surface_logic(script_number, unique_utts, surface_language=True)
        else:
            items = extract_entities_properties(merged_text, surface_language=False)
            prompt_content = prompt_surface_logic(script_number, items)
        
        prompt_content_total += prompt_content

    return merged_text_total, prompt_content_total

################## Logic to surface and sandwich

def permutation_sandwich_logic_surface_transl(input_text, substitutions_per_situation, sandwich_flag=None):

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
        substitutions = substitutions_per_situation.get(script_num, []) # Get the substitutions for this script number

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
            bot_utterances = {utt.strip() for utt in re.findall(bot_pattern, content) if utt.strip()}
            hum_utterances.update(bot_utterances)

        script_groups.setdefault(script_num, set()).update(hum_utterances)

    output_lines = []
    for script_num, utterances in script_groups.items():
        output_lines.append(f'<a script.{script_num} type={tag_type}>')
        output_lines.extend(f'<u speaker=HUM>{utt}</u>' for utt in utterances)
        output_lines.append('</a>\n')

    output_text = '\n'.join(output_lines)

    return output_text

