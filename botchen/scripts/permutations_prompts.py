# The script takes the data extracted from VG and applies words permutations from their synonyms or hypernyms. 
# These variations increase the generalization capability of Botchen.
# It also create a prompt file for automatic evaluation.

# **Usage:** ```python3 ./scripts/permutations_prompts.py```.
# Takes  **`./data/extracted_scripts.txt`** as input. Gives **`./data/training_data/*'** and **`./data/prompt_file.txt'** as output.
# The user can change the words to substitute at the end of the file.

import re
import os
import argparse

# Function to extract original situations from a reference file
def extract_situations(filename):
    with open(filename, "r", encoding="utf-8") as file:
        content = file.read()

    # Extract the scripts (situations) from the content using regular expressions
    scripts = re.findall(r"(<script\.\d+ type=CONV>.*?</script\.\d+>)", content, re.DOTALL)

    # Initialize a dictionary to store situations and count speaker occurrences
    situations = {}
    for i, script in enumerate(scripts, start=1):
        situations[f"Situation {i}"] = script
        speaker_count = len(re.findall(r"<u speaker=", script))
        print(f"Situation {i}, original script, has {speaker_count} utterances")

    return situations

# Function to apply word substitution and update script number
def substitute_word(script_text, old_word, new_word, new_script_number, script_number):
    modified_text = re.sub(rf'\b{re.escape(old_word)}\b', new_word, script_text)
    modified_text = modified_text.replace(f"script.{script_number}>", f"script.{script_number}.{new_script_number}>")
    modified_text = modified_text.replace(f"script.{script_number} type=CONV", f"script.{script_number}.{new_script_number} type=CONV")

    return modified_text

# Function to extract entities and properties from the script content
def extract_entities_properties(content):
    entity_properties = {}
    utterance_pattern = re.compile(r'<u speaker=[^>]*>\((.*?)\)</u>')
    utterances = utterance_pattern.findall(content)

    # Parse each utterance to separate entity and its properties
    for utterance in utterances:
        entity = utterance.split('.')[0]
        properties = utterance.split()[1:]
        properties = [re.sub(r'\.\d+', '', prop) for prop in properties]
        properties.insert(0, entity)
        if entity not in entity_properties:
            entity_properties[entity] = set()
        for prop in properties:
            entity_properties[entity].add(prop)
    
    # Return the number of entities, total properties, and the entity-property dictionary
    num_entities = len(entity_properties)
    num_properties = sum(len(properties) for properties in entity_properties.values())
    return num_entities, num_properties, entity_properties

# Function to generate a prompt script from the extracted entities and their properties
def generate_prompt_script_from_entities(script_number, entity_properties):
    script_content = f'<script.{script_number} type=CONV>\n'
    current_speaker = "HUM"
    
    # Iterate over each entity and add its properties to the script
    for entity, properties in entity_properties.items():
        filtered_properties = properties - {entity}
        properties_str = ' '.join(filtered_properties)
        script_content += f'<u speaker={current_speaker}>({entity}.n {properties_str})</u>\n'
        # Alternate between human and bot speakers
        if current_speaker == "HUM":
            current_speaker = "BOT"
        else:
            current_speaker = "HUM"

    script_content += f'</script.{script_number}>\n\n'
    return script_content

# Main function to apply substitutions and save modified scripts
def permutations(situations, substitutions_per_situation):
    new_situations = {}
    output_directory = "./data/training_data"
    os.makedirs(output_directory, exist_ok=True)
    all_count = 0
    
    # Process each situation and apply substitutions
    for script_id, script_text in situations.items():
        script_number = int(script_id.split()[1])
        print(script_number)
        substitutions = substitutions_per_situation.get(script_number, [])

        modified_text = script_text
        situation_versions = []
        for index, (old_word, new_word) in enumerate(substitutions, start=1):
            print(index)
            modified_text = substitute_word(modified_text, old_word, new_word, index, script_number)
            situation_versions.append(f"Situation {script_number}.{index}")
            new_situations[situation_versions[-1]] = modified_text
            
    # Merge and save the modified situations for training purposes
    merged_text_total = ''
    prompt_content_total = ''
    for script_number in range(1, 13):
        merged_text = ""
        for index in range(1, 10):
            situation_key = f"Situation {script_number}.{index}"
            if situation_key in new_situations:
                merged_text += new_situations[situation_key] + "\n\n"
                
        if merged_text:
            filename = os.path.join(output_directory, f"ideallanguage_{script_number}.txt")
            with open(filename, "w", encoding="utf-8") as file:
                file.write(merged_text)
                
            # Count the number of speakers and entities in the modified script
            speaker_count = len(re.findall(r"<u speaker=", merged_text))
            all_count += speaker_count
            print('')
            print(f"Situation {script_number}, with permutations, has {speaker_count} utterances")
            num_entities, num_properties, items = extract_entities_properties(merged_text)
            print('Entities:', num_entities, 'Properties', num_properties)
            merged_text_total += merged_text
            prompt_content = generate_prompt_script_from_entities(script_number, items)
            prompt_content_total += prompt_content
            
    # Print total number of utterances and entities across all situations
    print(f"\nAll situations together have {all_count} utterances")
    num_entities_total, num_properties_total, items_total = extract_entities_properties(merged_text_total)
    print('TRAINING Entities:', num_entities_total, 'Properties', num_properties_total, '\n')
    
    # Save the prompt content to a file for evaluation use
    if not os.path.exists('./data/prompt_file.txt'):
        with open('./data/prompt_file.txt', 'w') as file:
            file.write(prompt_content_total)
    else:
        with open('./data/prompt_file.txt', 'w') as file:
            file.write(prompt_content_total)
    # Print entity and property count for the prompt file
    num_entities_prompt, num_properties_prompt, items_prompt = extract_entities_properties(prompt_content_total)
    print('PROMPT Entities:', num_entities_prompt, 'Properties', num_properties_prompt, '\n')

# Main driver function to initialize the process
def main():

    # Extract original situations from the reference file
    situations = extract_situations("./data/extracted_scripts.txt")
    
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

    # Generate permutations and save the results
    permutations(situations, substitutions_per_situation)

# Execute the main function when the script is run
if __name__ == "__main__":
    main()

