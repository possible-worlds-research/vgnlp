import argparse
import pandas as pd
import numpy as np
import re
import os

directory = './data/chat/en/'

@vectorial_space.cli.command('create_overall_matrix')
# Create matrix without distinction between situations
def create_overall_matrix():
    entity_properties= {}
    # Loop over files
    for file_name in os.listdir(directory):
        if file_name.startswith('ideallanguage') and file_name.endswith('.txt'):
            file_path = os.path.join(directory, file_name)
            # Extract utterances
            with open(file_path, 'r') as file:
                poppi_content = file.read()
                utterances = re.findall(r'<u speaker=[^>]*>\((.*?)\)</u>', poppi_content)
                # Extract properties and entities
                for utterance in utterances:
                    entity = utterance.split('.')[0]
                    properties = utterance.split()[1:]
                    properties = [re.sub(r'\.\d+', '', prop) for prop in properties]
                    properties.insert(0, entity)
                    if entity not in entity_properties:
                        entity_properties[entity] = set()
                    for prop in properties:
                        entity_properties[entity].add(prop)
    # Tidy up
    all_properties = sorted(set(prop for props in entity_properties.values() for prop in props))
    entities = list(entity_properties.keys())
    activation_matrix = []

    for entity in entities:
        row = []
        for prop in all_properties:
            row.append(1 if prop in entity_properties[entity] else 0)
        activation_matrix.append(row)

    # Make up the matrix
    activation_matrix = np.array(activation_matrix)
    df = pd.DataFrame(activation_matrix, columns=all_properties, index=entities)

    # Store it
    df.to_csv('./data/chat/en/vectorial_dataframeCIAO.csv', index=True)
    print('Overall matrix created! :)')
    return activation_matrix, df

@vectorial_space.cli.command('create_script_matrix')
@click.argument('script_id')
def create_script_matrix(script_id):
    script_entity_properties = {}

    # Loop over the files
    for file_name in os.listdir(directory):
        if file_name.startswith('ideallanguage') and file_name.endswith('.txt'):
            file_path = os.path.join(directory, file_name)

            # Read the content of the file
            with open(file_path, 'r') as file:
                content = file.read()
                # Look at the situational script the user is interested in
                script_block = re.split(r'(<script\.\d+ type=CONV>)', content)
                script_pattern = r"<script." + str(script_id) + r"."
                # Iterate over blocks and process only the matching ones
                for block in script_block:
                    if script_pattern in block:
                        # Extract the utterances and extract also the entities and properties adding them to the dictionary
                        utterances = re.findall(r'<u speaker=[^>]*>\((.*?)\)</u>', block)
                        for utterance in utterances:
                            entity = utterance.split('.')[0]
                            properties = utterance.split()[1:]
                            properties = [re.sub(r'\.\d+', '', prop) for prop in properties]
                            properties.insert(0, entity)
                            if entity not in script_entity_properties:
                                script_entity_properties[entity] = set()
                            for prop in properties:
                                script_entity_properties[entity].add(prop)

    # Tidy up the properties and entities and make up the new activation matrix
    script_properties = sorted(set(prop for props in script_entity_properties.values() for prop in props))
    entities = list(script_entity_properties.keys())
    activation_matrix = []
    for entity in entities:
        row = []
        for prop in script_properties:
            row.append(1 if prop in script_entity_properties[entity] else 0)
        activation_matrix.append(row)
    df = pd.DataFrame(activation_matrix, columns=script_properties, index=entities)
    print('The script matrix has been created! :)')
    return activation_matrix, df

########################################

from os.path import dirname, realpath, join
from glob import glob
from flask import Blueprint
import click
import torch

from app import app
from app.cli.prepare import preprocess
from app.cli.statlm import train_stat_lm
from app.cli.train import run_training, ctx
from app.loaders import load_gpt_models
from app.chat.converse import generate
from app.utils import load_models_and_hyperparameters

from app import log_path
from datetime import datetime
from flask import session
from app.utils import split_by_space
from app.chat.memory_processor import write_to_short_term_memory
import os
import re
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from app.cli.vectorial_space import create_overall_matrix, create_script_matrix

torch.set_num_threads(1)

training = Blueprint('training', __name__)

dir_path = dirname(dirname(dirname(realpath(__file__))))
modules_path = join(dir_path, 'app', 'modules')
data_path = join(dir_path, 'data')

@vectorial_space.cli.command('evaluation_with_vectorial_space')
@click.argument('module')
@click.argument('language')
@click.argument('topk')
# With this script I am creating a vectorial space from the responses the model gives to prompts from the training data in order to be abl>
def evaluation_with_vectorial_space(module, language, topk):

    # Which model are we using?
    gpt_models = load_gpt_models(module, modules_path, language)
    model = gpt_models['chat'][0]
    encoder = gpt_models['chat'][1]
    decoder = gpt_models['chat'][3]
    print(f">> Loaded models {gpt_models.keys()}\n\n")

    # User information for saving
    conversation_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs/evaluation/'))
    log_file = os.path.join(log_dir, f'evaluation_user-{conversation_time}.txt')
    with open(log_file, 'a') as file:
        file.write(f">> Loaded models {gpt_models.keys()}\n\n")

    # Store the conversation
    conversation = ''
    # For evaluation
    correct_responses = 0
    total_responses = 0
    # Storing and recording
    response_history = {}
    # Store to evaluate with the vectorial space
    entity_properties= {}

    # Read the reference file with the utterances and extract the prompts
    with open('./data/chat/en/reference_script.txt', 'r') as file:
        poppi_content = file.read()
        utterances = re.findall(r'<u speaker=[^>]*>\((.*?)\)</u>', poppi_content)

    # Go trough the utterances and prompt them
    for utterance in utterances:
        response_history[utterance] = []
        # Prompt 4 times the same utterance
        for _ in range(4):
            # Make a conversation that the model can read
            if len(conversation.strip()) > 0:
                conversation += f' <u speaker=HUM>{utterance}</u> <u speaker=BOT>'
            else:
                conversation = f'<u speaker=HUM>{utterance}</u> <u speaker=BOT>'

            # Make the conversation start
            with torch.no_grad():
                with ctx:
                    response, _ = generate(utterance, conversation, model, encoder, decoder, 0.9, int(topk), 64)
            # Clean the responses so to evaluate them
            match = re.search(r'\((.*?)\)', response)
            if match:
                clean_response = match.group(1)
            else:
                print('Hey, too weird output to extract maybe (?) :(', response)
                continue

            # Printing statements and saving to recording file
            print(f">> Prompt: {utterance}\n>> Response: {response} \n\n")
            with open(log_file, 'a') as file:
                log_entry = f"\n>> Prompt: {utterance}\n>> Response: {response} \n"
                file.write(log_entry)

            # Extracting the entity and the properties the model is outputting
            entity = clean_response.split('.')[0]
            properties = clean_response.split()[1:]
            properties = [re.sub(r'\.\d+', '', prop) for prop in properties]
            properties.insert(0, entity)

        # Store them in entity_properties for evaluation
            if entity not in entity_properties:
                entity_properties[entity] = set()
            for prop in properties:
                entity_properties[entity].add(prop)

    # Keeping the structure of the training-based semantic space (entities and properties in the structure that they are already), create >
    responses_df = pd.read_csv('./data/chat/en/vectorial_dataframe.csv', index_col=0)
    responses_df[:] = 0

    # Make up the new dataframe with the new data
    for entity, props in entity_properties.items():
        if entity in responses_df.index:  # Ensure the entity exists in the original dataframe
            for prop in props:
                if prop in responses_df.columns:  # Ensure the property exists in the original columns
                    responses_df.loc[entity, prop] = 1 # Set the cell to 1
    responses_matrix = responses_df.values
    print('NEW VECTORIAL SPACE FROM OUTPUTS \n\n', responses_df)
    responses_df.to_csv('./data/chat/en/vectorial_dataframe_evaluation.csv', index=True)

    # Reload the previous matrix
    optimal_df = pd.read_csv('./data/chat/en/vectorial_dataframe.csv', index_col=0)
    optimal_matrix = optimal_df.values

    # Compare the two spaces (optimal and response based one)
    similarity_matrix = cosine_similarity(optimal_matrix, responses_matrix)
    similarity_df = pd.DataFrame(similarity_matrix, index=optimal_df.index, columns=responses_df.index)
    print('COSINE SIMILARITY MATRIX \n\n', similarity_df)

    # Possible numeracy value for accuracy
    diagonal_similarity = np.diagonal(similarity_matrix)
    average_similarity = np.mean(diagonal_similarity)
    print(f'Average cosine similarity between the training-based and the response-based spaces: {average_similarity:.4f}')
    # for idx, entity in enumerate(optimal_df.index):
    #    print(f"Similarity for entity {entity}: {diagonal_similarity[idx]:.4f}")
    
@vectorial_space.cli.command('evaluation_with_vectorial_space_trial')
@click.argument('module')
@click.argument('language')
@click.argument('topk')
@click.argument('script_id')
# With this script I am creating a vectorial space from the responses the model gives to prompts from the training data in order to be abl>
def evaluation_with_vectorial_space_contextual(module, language, topk, script_id):

    # Store the conversation
    conversation = ''
    # For evaluation
    correct_responses = 0
    total_responses = 0
    # Storing and recording
    # response_history = {}
    # Store to evaluate with the vectorial space
    entity_properties= {}

    # MODEL used
    gpt_models = load_gpt_models(module, modules_path, language)
    model = gpt_models['chat'][0]
    encoder = gpt_models['chat'][1]
    decoder = gpt_models['chat'][3]
    print(f">> Loaded models {gpt_models.keys()}\n\n")

    with open('./data/chat/en/reference_script.txt', 'r') as file:
        content = file.read()

    # Look at the situational script the user is interested in
    script_blocks = re.split(r'(<script\.\d+ type=CONV>)', content)
    script_dict = {script_blocks[i]: script_blocks[i + 1] for i in range(1, len(script_blocks) - 1, 2)}
    script_pattern = f"script.{script_id} "
    print('SCRIPT_PATTERN', script_pattern)
    for header, content in script_dict.items():
        if script_pattern in header:
            print('HEADER:', header)
            print('CONTENT:', content)
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

    # CREATING THE RESPONSE MATRIX
    all_properties = sorted(set(prop for props in entity_properties.values() for prop in props))
    entities = list(entity_properties.keys())
    eval_matrix = []

    for entity in entities:
        row = []
        for prop in all_properties:
            row.append(1 if prop in entity_properties[entity] else 0)
        eval_matrix.append(row)
    eval_df = pd.DataFrame(eval_matrix, columns=all_properties, index=entities)
    print('RESPONSES_DF', eval_df)

    # Keeping the structure of the training-based semantic space (entities and properties in the structure that they are already), create >
    optimal_matrix, optimal_df = create_script_matrix(script_id)
    # optimal_matrix, optimal_df = create_script_matrix(script_id)
    print('ORIGINAL_DF', optimal_df)

    # OR, FOR THE NEW DATAFRAME, USE THE STRUCTURE WHICH ALREADY EXISTS
    # responses_df = original_script_df
    # for entity, props in entity_properties.items():
    #     if entity in responses_df.index:  # Ensure the entity exists in the original dataframe
    #         for prop in props:
    #             if prop in responses_df.columns:  # Ensure the property exists in the original columns
    #                 responses_df.loc[entity, prop] = 1 # Set the cell to 1
    # responses_matrix = responses_df.values
    # print('NEW VECTORIAL SPACE FROM OUTPUTS \n\n', responses_df)

    # Aligning U' (responses-based) to U (original)

    # matching_rows = eval_df.index.intersection(optimal_df.index)
    # matching_cols = eval_df.columns.intersection(optimal_df.columns)

    # Match the dimensions (fill out with zeros)
    aligned_evaluation_df = eval_df.reindex(index=optimal_df.index, fill_value=0)
    aligned_evaluation_df = aligned_evaluation_df.reindex(columns=optimal_df.columns, fill_value=0)
    print('ALIGNED EVALUATION', aligned_evaluation_df)
    # Compute the  aggregation function (along rows)
    aggregated_matrix = aligned_evaluation_df.sum(axis=0).to_frame().T  # Aggregating along columns
    aggregated_matrix = aggregated_matrix.reindex(columns=optimal_df.columns, fill_value=0)  # Align columns with optimal_df
    print('AGRREGATED_MATRIX', aggregated_matrix)
    similarity_matrix = cosine_similarity(optimal_df.to_numpy(), aggregated_matrix.to_numpy())
    similarity_df = pd.DataFrame(similarity_matrix, index=optimal_df.index, columns=aggregated_matrix.index)
    print('SIMILARITY_df', similarity_df)
    average_similarity = np.mean(similarity_matrix)
    print('AVERAGE', average_similarity)

    # missing_rows = eval_df.index.difference(optimal_df.index)
    # missing_cols = eval_df.columns.difference(optimal_df.columns)
    # print('MISSING ROWS', missing_rows, '\nMISSING COLUMNS', missing_cols)

    # aligned_eval_df = eval_df.loc[matching_rows, matching_cols]
    # aggregated_matrix = aligned_eval_df.sum(axis=0).to_frame().T  # Sum over features
    # aggregated_matrix = aggregated_matrix.reindex(columns=optimal_df.columns, fill_value=0)
    # print('AGGREGATED MATRIX', aggregated_matrix)

    # similarity_matrix = cosine_similarity(optimal_df.to_numpy(), aggregated_matrix.to_numpy())
    # similarity_df = pd.DataFrame(similarity_matrix, index=optimal_df.index, columns=["A(U')"])
    # average_similarity = np.mean(similarity_matrix)