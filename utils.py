
################# Utils

def extract_entities_properties(content, entity_properties):
    utterance_pattern = re.compile(r'<u speaker=[^>]*>\((.*?)\)</u>')
    utterances = utterance_pattern.findall(content)

    for utterance in utterances:
        entity = utterance.split('.')[0]
        properties = utterance.split()[1:]
        properties = [re.sub(r'\.\d+', '', prop) for prop in properties]
        properties.insert(0, entity)
        if entity not in entity_properties:
            entity_properties[entity] = set()
        for prop in properties:
            entity_properties[entity].add(prop)

def make_evaluating_conversation(content, # From which to extract the utterances
                                 conversation, # Conversation made 'til that point
                                 entity_properties, # Specificities of script storing
                                 model, encoder, decoder, topk, # Needed since I am calling the model live
                                 log_file = False, # If we want to store
                                 print_statement = False): # If we want to look live at the conversation
    utterances = re.findall(r'<u speaker=[^>]*>\((.*?)\)</u>', content)
    for utterance in utterances:
        # CONVERSATIONAL PURPOSES
        for _ in range(4):
            if len(conversation.strip()) > 0:
                conversation += f' <u speaker=HUM>{utterance}</u> <u speaker=BOT>'
            else:
                conversation = f'<u speaker=HUM>{utterance}</u> <u speaker=BOT>'
            with torch.no_grad():
                with ctx:
                    response, _ = generate(utterance, conversation, model, encoder, decoder, 0.9, int(topk), 64)

            # CLEAN UP THE MATERIAL
            match = re.search(r'\((.*?)\)', response)
            if match:
                clean_response = match.group(1)
            else:
                print('TOO WEIRD TO EXTRACT :((', response)
                continue

            # EXTRACT ENTITIES AND PROPERTIES
            entity = clean_response.split('.')[0]
            properties = clean_response.split()[1:]
            properties = [re.sub(r'\.\d+', '', prop) for prop in properties]
            properties.insert(0, entity)
            if entity not in entity_properties:
                entity_properties[entity] = set()
            for prop in properties:
                entity_properties[entity].add(prop)

            if print_statement:
                print(f">> Prompt: {utterance}\n>> Response: {response} \n\n")
            if log_file:
                with open(log_file, 'a') as file:
                    log_entry = f"\n>> Prompt: {utterance}\n>> Response: {response} \n"
                    file.write(log_entry)
