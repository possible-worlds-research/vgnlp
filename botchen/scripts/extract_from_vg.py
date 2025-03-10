import re
import os
import zipfile
import argparse

def extract(file_path, situation_id, new_situation_id):
    with open(file_path, 'r') as file:
        content = file.read()

    # Extract <situation> block
    situation_pattern = rf"<situation id={situation_id}>(.*?)</situation>"
    match = re.search(situation_pattern, content, re.DOTALL)
    if not match:
        print(f"No situation found with id={situation_id}")
        return None

    situation_content = match.group(1)

    # Dictionaries to store entity data
    entity_map = {}  # Maps entity ID -> entity name
    properties_map = {}  # Maps entity ID -> list of properties

    # Process each <entity> block
    entity_pattern = re.compile(r"<entity id=(\d+)>\s*(.*?)\s*</entity>", re.DOTALL)
    for entity_match in entity_pattern.finditer(situation_content):
        entity_id = entity_match.group(1)
        entity_lines = entity_match.group(2).strip().split("\n")

        # Extract entity name (first line)
        first_line = entity_lines[0].strip()
        entity_name_match = re.match(r"([\w\.]+)\(\d+\)", first_line)
        if entity_name_match:
            entity_name = entity_name_match.group(1)
            # Keep ".n" in the entity name, but remove the numeric suffix (i.e., .n.xx)
            clean_entity_name = re.sub(r"\.n\.\d+", ".n", entity_name)
            entity_map[entity_id] = clean_entity_name
            properties_map[entity_id] = []

    # Second pass: Extract properties and relations
    for entity_match in entity_pattern.finditer(situation_content):
        entity_id = entity_match.group(1)
        entity_lines = entity_match.group(2).strip().split("\n")

        for line in entity_lines[1:]:
            line = line.strip()
            prop_match = re.match(r"([\w|]+)\(([\d,]+)\)", line)
            if prop_match:
                prop_name, prop_ids = prop_match.groups()
                prop_ids = prop_ids.split(",")

                # If it's a relation (references another entity)
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

    script_content = f"<script.{new_situation_id} type=CONV>\n"
    current_speaker = "HUM"
    for entity_id, entity_name in entity_map.items():
        clean_entity_name = entity_name
        properties_str = " ".join(re.sub(r"\.n\.\d+", "", prop) for prop in properties_map[entity_id])
        script_content += f'<u speaker={current_speaker}>({clean_entity_name} {properties_str})</u>\n'

        if current_speaker == "HUM":
            current_speaker = "BOT"
        else:
            current_speaker = "HUM"

    script_content += f"</script.{new_situation_id}>\n\n"
    return script_content

def main(file_path):

    text1 = extract(file_path, 1, 1)
    text2 = extract(file_path, 2, 2)
    text3 = extract(file_path, 4, 3)
    text4 = extract(file_path, 71, 4)
    text5 = extract(file_path, 9, 5)
    text6 = extract(file_path, 2410753, 6)
    text7 = extract(file_path, 713137, 7)
    text8 = extract(file_path, 2412620, 8)
    text9 = extract(file_path, 2412211, 9)
    text10 = extract(file_path, 186, 10)
    text11 = extract(file_path, 2396154, 11)
    text12 = extract(file_path, 2317468, 12)

    texts = [text1, text2, text3, text4, text5, text6, text7, text8, text9, text10, text11, text12]
    merged_text = "".join(texts)

    with open('./data/extracted_scripts.txt', "w", encoding="utf-8") as file:
        file.write(merged_text)

if __name__ == "__main__":
    main("./data/ideallanguage.txt")

