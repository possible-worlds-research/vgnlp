# The script takes the data extracted from VG and applies words permutations from their synonyms or hypernyms. 
# These variations increase the generalization capability of Botchen.
# It also create a prompt file for automatic evaluation.

# **Usage:** ```python3 ./scripts/permutations_prompts.py```.
# Takes  **`./data/extracted_scripts.txt`** as input. Gives **`./data/training_data/*'** and **`./data/prompt_file.txt'** as output.
# The user can change the words to substitute at the end of the file.

import re
import os
import argparse
import random
import nltk
from nltk.corpus import wordnet as wn
from collections import defaultdict
import nltk
from nltk.translate.bleu_score import sentence_bleu

# Download required resources
# nltk.download('wordnet')
# nltk.download('omw-1.4')  # For wordnet synonyms in different languages

################# Logic to logic and Surface to surface

# Thoughts: This could also be done in relation to other entities without retrieving the files

# Function to extract original situations from a reference file
def extract_situations(filename):
    with open(filename, "r", encoding="utf-8") as file:
        content = file.read()
    scripts = re.findall(r"(<script\.\d+ type=CONV>.*?</script\.\d+>)", content, re.DOTALL)
    # Initialize a dictionary to store situations and count speaker occurrences
    situations = {}
    for i, script in enumerate(scripts, start=1):
        situations[f"Situation {i}"] = script
        speaker_count = len(re.findall(r"<u speaker=", script))
        print(f"{filename}: Situation {i}, original script, has {speaker_count} utterances")
    return situations

# Function to apply word substitution and update script number
def substitute_word(script_text, old_word, new_word):
    modified_text = re.sub(rf'\b{re.escape(old_word)}\b', new_word, script_text)
    return modified_text if modified_text != script_text else None

# Thoughts: This we could also make it interacting in other ways with the entities

# Function to extract entities and properties from the script content
def extract_entities_properties(content, surface_language = False):
    if surface_language:
        utterances = re.findall(r'<u speaker=[^>]*>(.*?)</u>', content)
        return list(set(utterances))
    entity_properties = {}
    utterances = re.findall(r'<u speaker=[^>]*>\((.*?)\)</u>', content)
    for utterance in utterances:
        entity, properties = utterance.split()
        properties = [re.sub(r'\.\d+', '', prop) for prop in properties]
        if entity not in entity_properties:
            entity_properties[entity] = set()
        entity_properties[entity].update(properties)
    return entity_properties

# Function to generate a prompt script from the extracted entities and their properties
def generate_prompt_script_from_entities(script_number, entity_properties, surface_language = False):
    script_content = f'<script.{script_number} type=CONV>\n'
    if surface_language:
        for utterance in entity_properties:
            script_content += f'<u speaker=HUM>{utterance}</u>\n'
    else:
        for entity, properties in entity_properties.items():
            filtered_properties = properties - {entity}
            properties_str = ' '.join(filtered_properties)
            script_content += f'<u speaker=HUM>({entity}.n {properties_str})</u>\n'
    script_content += f'</script.{script_number}>\n\n'
    return script_content

# Main function to apply substitutions and save modified scripts
def permutations(situations, substitutions_per_situation, surface_language=False):

    new_situations = {}
    output_directory = "./data/training"
    os.makedirs(output_directory, exist_ok=True)
    all_count = 0

    # Process each situation and apply substitutions
    for script_id, script_text in situations.items():
        script_number = int(script_id.split()[1])
        substitutions = substitutions_per_situation.get(script_number, [])
        situation_versions = []
        for index, (old_word, new_word) in enumerate(substitutions, start=1):
            modified_text = substitute_word(script_text, old_word, new_word)
            if modified_text:  # Only update and store if substitution occurred
                version_name = f"Situation {script_number}.{index}"
                situation_versions.append(version_name)
                new_situations[version_name] = modified_text

    # Merge and save the modified situations for training purposes
    merged_text_total = ''
    prompt_content_total = ''
    for script_number in range(1, 13): # Thoughts: this should be changed to be dynamical
        merged_text = ""
        for index in range(1, 10):
            situation_key = f"Situation {script_number}.{index}"
            if situation_key in new_situations:
                merged_text += new_situations[situation_key] + "\n\n"

        if merged_text:
            # Count the number of speakers and entities in the modified script
            speaker_count = len(re.findall(r"<u speaker=", merged_text))
            all_count += speaker_count
            if surface_language:
                unique_utterances, num_utterances = extract_entities_properties(merged_text, surface_language=True)
                prompt_content = generate_prompt_script_from_entities(script_number, unique_utterances, surface_language=True)
                prompt_content_total += prompt_content
            else:
                num_entities, num_properties, items = extract_entities_properties(merged_text, surface_language=False)
                prompt_content = generate_prompt_script_from_entities(script_number, items)
                prompt_content_total += prompt_content
            merged_text_total += merged_text

    filename = os.path.join(output_directory, "permuted_surface.txt" if surface_language else "permuted_logical.txt")
    with open(filename, "w", encoding="utf-8") as file:
        file.write(merged_text_total)
    if surface_language:
        with open('./data/training/prompt_surface.txt', 'w', encoding='utf-8') as file:
            file.write(prompt_content_total)
    else:
        with open('./data/training/prompt_logical.txt', 'w', encoding='utf-8') as file:
            file.write(prompt_content_total)

################## Logic to surface and sandwich

# Function to apply word substitution and update script number
def apply_substitutions(text, substitutions):
    original_text = text
    for old, new in substitutions:
        if old in text:
            new_text = re.sub(r'\b' + re.escape(old) + r'\b', new, text)  # Replace only whole words
    return text if text != original_text else None

# Thoughts: They should be optimized and also I think these could be merged with the others
# Function to apply word substitution and update script number
def apply_substitutions_sandwich(hum_text, bot_text_one, bot_text_two, bot_text_three,  substitutions):
    new_hum_text = hum_text
    new_bot_text_one = bot_text_one
    new_bot_text_two = bot_text_two
    new_bot_text_three = bot_text_three

    for old, new in substitutions:
        if old:
            if old in new_hum_text:
                new_hum_text = re.sub(r'\b' + re.escape(old) + r'\b', new, new_hum_text)
            if old in new_bot_text_one:
                new_bot_text_one = re.sub(r'\b' + re.escape(old) + r'\b', new, new_bot_text_one)
            if old in new_bot_text_two:
                new_bot_text_two = re.sub(r'\b' + re.escape(old) + r'\b', new, new_bot_text_two)
            if old in new_bot_text_three:
                new_bot_text_three = re.sub(r'\b' + re.escape(old) + r'\b', new, new_bot_text_three)

    if (new_hum_text == hum_text and new_bot_text_one == bot_text_one and new_bot_text_two == bot_text_two and new_bot_text_three == bot_text_three):
        return None, None, None, None

    return (
        new_hum_text if new_hum_text != hum_text else hum_text,
        new_bot_text_one if new_bot_text_one != bot_text_one else bot_text_one,
        new_bot_text_two if new_bot_text_two != bot_text_two else bot_text_two,
        new_bot_text_three if new_bot_text_three != bot_text_three else bot_text_three
    )

def process_text(input_path, substitutions_per_situation, output_path, sandwich_flag=None):
    with open(input_path) as file:
        input_text=file.read()

    pattern = r'<a script\.(\d+) type=DSC>\s*<u speaker=HUM>(.*?)</u>\s*<u speaker=BOT>(.*?)</u>\s*</a>'
    if sandwich_flag:
        pattern = r'<a script\.(\d+) type=SDW>\s*<u speaker=HUM>(.*?)</u>\s*<u speaker=BOT>(.*?)</u>\s*<u speaker=BOT>(.*?)</u>\s*<u speaker=BOT>(.*?)</u>\s*</a>'

    matches = re.findall(pattern, input_text, re.DOTALL)
    processed_text = ""

    previous_script_number= None

    for match in matches:
        script_num = int(match[0])  # Get the script number
        hum_text = match[1]  # Get the human text
        bot_text = match[2]  # Get the bot text
        if sandwich_flag:
            bot_text_one = match[2]  # Get the bot text
            bot_text_two = match[3]  # Get the bot text
            bot_text_three = match[4]  # Get the bot text

        # Get the substitutions for this script number
        substitutions = substitutions_per_situation.get(script_num, [])

        # Apply the substitutions and get both original and substituted text
        new_hum = apply_substitutions(hum_text, substitutions)
        new_bot = apply_substitutions(bot_text, substitutions)
        if sandwich_flag:
            new_hum, new_bot_one, new_bot_two, new_bot_three = apply_substitutions_sandwich(hum_text, bot_text_one, bot_text_two, bot_text_three, substitutions)

        if previous_script_number is not None and script_num != previous_script_number:
            processed_text += ""

        if not sandwich_flag:
            processed_text += f"<a script.{script_num} type=DSC>\n"
            processed_text += f"<u speaker=HUM>{hum_text}</u>\n"
            processed_text += f"<u speaker=BOT>{bot_text}</u>\n"
        if sandwich_flag:
            processed_text += f"<a script.{script_num} type=SDW>\n"
            processed_text += f"<u speaker=HUM>{hum_text}</u>\n"
            processed_text += f"<u speaker=BOT>{bot_text_one}</u>\n"
            processed_text += f"<u speaker=BOT>{bot_text_two}</u>\n"
            processed_text += f"<u speaker=BOT>{bot_text_three}</u>\n"
        processed_text += "</a>\n\n"

        if new_hum and new_bot:  # If substitution occurred
            if not sandwich_flag:
                processed_text += f"<a script.{script_num} type=DSC>\n"
                processed_text += f"<u speaker=HUM>{new_hum}</u>\n"
                processed_text += f"<u speaker=BOT>{new_bot}</u>\n"
            if sandwich_flag:
                processed_text += f"<a script.{script_num} type=SDW>\n"
                processed_text += f"<u speaker=HUM>{new_hum}</u>\n"
                processed_text += f"<u speaker=BOT>{new_bot_one}</u>\n"
                processed_text += f"<u speaker=BOT>{new_bot_two}</u>\n"
                processed_text += f"<u speaker=BOT>{new_bot_three}</u>\n"
            processed_text += "</a>\n\n"

        previous_script_number = script_num

    with open(output_path, 'w') as file:
        file.write(processed_text)
    return processed_text

# Function to extract unique HUM utterances per script number
def extract_unique_hum_utterances(input_text, output_path, sandwich_flag=None):
    # Regex to match <a> blocks for script numbers and their HUM/BOT utterances
    pattern = r'<a script\.(\d+) type=DSC>(.*?)</a>'
    if sandwich_flag:
        pattern = r'<a script\.(\d+) type=SDW>(.*?)</a>'

    # Find all <a> blocks and group them by script number
    script_groups = {}

    matches = re.findall(pattern, input_text, re.DOTALL)
    for match in matches:
        script_num = match[0]
        content = match[1]

        # Extract all HUM utterances
        if not sandwich_flag:
            hum_pattern = r'<u speaker=HUM>(.*?)</u>'
            hum_utterances = re.findall(hum_pattern, content)
        if sandwich_flag:
            hum_pattern_one = r'<u speaker=HUM>(.*?)</u>'
            hum_pattern_two = r'<u speaker=BOT>([^()<>\n]+)</u>'
            hum_utterances_one = re.findall(hum_pattern_one, content)
            hum_utterances_two = re.findall(hum_pattern_two, content)
            hum_utterances_two = [utt.strip() for utt in hum_utterances_two if utt.strip()]

        # Add the unique HUM utterances to the script_groups dictionary
        if script_num not in script_groups:
            script_groups[script_num] = set()
        if not sandwich_flag:
            script_groups[script_num].update(hum_utterances)
        if sandwich_flag:
            script_groups[script_num].update(hum_utterances_one)
            script_groups[script_num].update(hum_utterances_two)

    # Prepare the final output with unique HUM utterances
    output_text = ""

    for script_num, hum_utterances in script_groups.items():
        if not sandwich_flag:
            output_text += f'<a script.{script_num} type=DSC>\n'
        if sandwich_flag:
            output_text += f'<a script.{script_num} type=SDW>\n'
        for hum in hum_utterances:
            output_text += f'<u speaker=HUM>{hum}</u>\n'
        output_text += '</a>\n\n'

    with open(output_path, 'w') as file:
        file.write(output_text)

    return output_text

################# Main function

def main():

    # Define word substitutions for each situation
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

    permutations(extract_situations("./data/training/extracted_logical.txt"), substitutions_per_situation, surface_language=False)

    permutations(extract_situations('./data/training/extracted_surface.txt'),substitutions_per_situation, surface_language=True)

    extract_unique_hum_utterances(process_text('./data/training/extracted_logical_to_surface.txt', substitutions_per_situation, './data/training/permuted_logical_to_surface.txt'), './data/training/prompt_logical_to_surface.txt')

    extract_unique_hum_utterances(process_text('./data/training/extracted_surface_to_logical.txt', substitutions_per_situation, './data/training/permuted_surface_to_logical.txt'), './data/training/prompt_surface_to_logical.txt')

    extract_unique_hum_utterances(process_text('./data/training/extracted_sandwich.txt', substitutions_per_situation, './data/training/permuted_sandwich.txt', sandwich_flag=1), './data/training/prompt_sandwich.txt', sandwich_flag=1)

if __name__ == "__main__":
    main()

################### UTILS

def get_wordnet_hypernyms(entity):
    noun_hypernyms = set()

    # Collect hypernyms for the entity
    for lemma in wn.lemmas(entity):
        for hypernym in lemma.synset().hypernyms():
            # Only consider hypernyms that are nouns (part-of-speech 'n')
            if hypernym.pos() == 'n':
                noun_hypernyms.add(hypernym.name().split('.')[0])  # Remove the '.n.01' part
    hypernym = list(noun_hypernyms)[:1]
    # Return the noun hypernyms as a list
    return hypernym

def define_substitutions(x, situations):
    substitutions_per_situation = defaultdict(list)

    for script_id, script_text in situations.items():
        script_number = int(script_id.split()[1])
        # Extract entities from the script text
        num_entities, num_properties, items = extract_entities_properties(script_text)
        entities = list(items.keys())  # Get the entities (keys of the extracted properties)

        selected_entities = random.sample(entities, x)
        print(f"Selected entities for script {script_number}: {selected_entities}")

        substitutions = []  # List to store substitutions for the current script
        for entity in selected_entities:
            # Get hypernyms for the entity
            hypernyms = get_wordnet_hypernyms(entity)
            print(f"For Entity: {entity}, Hypernyms: {hypernyms}")

            # Initialize substitution for each entity with a placeholder
            entity_substitution = [('', '')]

            if hypernyms:
                entity_substitution.append((entity, hypernyms[0]))  # Add the first hypernym if available

            substitutions.append(entity_substitution)

        substitutions_per_situation[script_number] = substitutions

    print(dict(substitutions_per_situation))
    return dict(substitutions_per_situation)