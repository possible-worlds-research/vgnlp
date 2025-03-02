from os.path import dirname, realpath, join
from flask import Blueprint
import click
import torch

from app.cli.prepare import preprocess
from app.cli.statlm import train_stat_lm
from app.cli.train import run_training, ctx
from app.loaders import load_gpt_models
from app.chat.converse import generate
from app.utils import split_by_space

# from utils import extract_entities_properties, make_evaluating_conversation

from datetime import datetime
import pandas as pd
import numpy as np
import re
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

############################ PREVIOUS CODE

torch.set_num_threads(1)

training = Blueprint('training', __name__)

dir_path = dirname(dirname(dirname(realpath(__file__))))
modules_path = join(dir_path, 'app', 'modules')
data_path = join(dir_path, 'data')

@training.cli.command('run')
@click.argument('module')
@click.argument('language')
@click.argument('vocab_size')
def train(module, language, vocab_size):
    data_dir = join(data_path, module, language)
    module_dir = join(modules_path, module, language)
    preprocess(data_dir, module_dir, vocab_size)
    train_stat_lm(data_dir, module_dir)
    run_training(data_dir, module_dir)

@training.cli.command('eval')
@click.argument('module')
@click.argument('language')
@click.argument('topk')
def chat(module, language, topk):
    gpt_models = load_gpt_models(module, modules_path, language)
    model = gpt_models['chat'][0]
    encoder = gpt_models['chat'][1]
    decoder = gpt_models['chat'][3]
    print(f">> Loaded models {gpt_models.keys()}\n\n")
    prompt = ''
    conversation = ''
    while True:
        prompt = input('>> ').rstrip('\n')
        if prompt == 'q':
            break
        if len(conversation.strip()) > 0:
            conversation = conversation + ' <u speaker=HUM>'+prompt+'</u> <u speaker=BOT>'
        else:
            conversation = '<u speaker=HUM>'+prompt+'</u> <u speaker=BOT>'
        with torch.no_grad():
            with ctx:
                response, _ = generate(prompt, conversation, model, encoder, decoder, 0.9, int(topk), 64)
                print('>> '+response)
                conversation = conversation + response + '</u>'
    conversation_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs/'))
    log_file = os.path.join(log_dir, f'user-{conversation_time}.txt')
    with open(log_file,'a') as fout:
        #print(split_by_space(conversation)[-2:])
        fout.write('\n'.join(split_by_space(conversation)[-2:])+'\n')

######################################## UTILS

directory = './data/chat/en/'
saving_directory = './data/chat/en/vectorial_space.csv'

# @training.cli.command('create_matrices')
# @click.argument('script_id', type = int)
# @click.argument('optimal_matrix', type=int)
# @click.argument('entity_properties_dynamical',required=False)
# @click.argument('store_space', type=int, required=False)
# Create matrix without distinction between situations
def create_matrices(script_id, # If we want to make a matrix from a specific situation, which one? If it is not there it retrieves the all data
                    optimal_matrix, # Optimal means retrieving from training data, non optimal from Botchen conversation
                    entity_properties_dynamical, # If from Botchen conversation, the entity and properties are dynamical
                    store_space):

    if optimal_matrix == 1: # Retrieve directly from training data
        entity_properties= {}
        # Loop over files
        for file_name in os.listdir(directory):
            if file_name.startswith('ideallanguage') and file_name.endswith('.txt'):
                file_path = os.path.join(directory, file_name)
                # Extract utterances
                with open(file_path, 'r') as file:
                    content = file.read()
                if script_id != 0: # Create a situational matrix
                    script_block = re.split(r'(<script\.\d+ type=CONV>)', content)
                    script_pattern = r"<script." + str(script_id) + r"."
                    # Iterate over blocks and process only the matching ones
                    for block in script_block:
                        if script_pattern in block:
                            extract_entities_properties(block, entity_properties)
                if script_id == 0: # Create the matrix from the full space
                    extract_entities_properties(content, entity_properties)

    else: # Create a matrix from the conversation with Botchen for evaluation
        entity_properties = entity_properties_dynamical

    # Retrieve properties and entities and create the matrix
    all_properties = sorted(set(prop for props in entity_properties.values() for prop in props))
    entities = list(entity_properties.keys())
    activation_matrix = []
    for entity in entities:
        row = [1 if prop in entity_properties[entity] else 0 for prop in all_properties]
        activation_matrix.append(row)
    df = pd.DataFrame(activation_matrix, columns=all_properties, index=entities)

    if optimal_matrix == 1:
        print('Optimal matrix has been created')
    if optimal_matrix == 0:
        print('Matrix from Botchen conversation has been created')
    if script_id != 0:
        print('Matrix from the specific situation has been created')
    if script_id == 0:
        print('Matrix from the full corpus has been created')

    if store_space == 1: # Store the matrix?
        df.to_csv(saving_directory, index=True)
        print('Matrix has been stored as {saving_directory}')

    return activation_matrix, df

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

################################## ACTUAL EVALUATION CODE

@training.cli.command('evaluation_with_vectorial_space')
@click.argument('module')
@click.argument('language')
@click.argument('topk')
@click.argument('evaluating_framework', required=False, type=int) # Which kind of evaluating framework do we want?
@click.argument('script_id_eval', type=int) # Which script to evaluate
@click.argument('script_id_optimal', type=int) # With which other one
@click.argument('store_conversation', required=False,type=int)
# With this script I am creating a vectorial space from the responses the model gives to prompts from the training data in order to be abl>
# If evaluating on a situational script, see if to point the evaluation script on one situation and and if also to use the situation for the optimal one
def evaluation_with_vectorial_space(module, language, topk,
                                    evaluating_framework,
                                    script_id_eval,
                                    script_id_optimal,
                                    store_conversation):

    # Which model are we using?
    gpt_models = load_gpt_models(module, modules_path, language)
    model = gpt_models['chat'][0]
    encoder = gpt_models['chat'][1]
    decoder = gpt_models['chat'][3]
    print(f">> Loaded models {gpt_models.keys()}\n\n")

    if store_conversation == 1:
        # User information for saving
        conversation_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs/evaluation/'))
        log_file = os.path.join(log_dir, f'evaluation_user-{conversation_time}.txt')
        with open(log_file, 'a') as file:
            file.write(f">> Loaded models {gpt_models.keys()}\n\n")

    conversation = ''
    entity_properties= {}

    # Read the reference file with the utterances and extract the prompts
    with open('./data/chat/en/reference_script.txt', 'r') as file:
        content = file.read()

    if script_id_eval != 0: # If evaluating on only one script situation
        script_blocks = re.split(r'(<script\.\d+ type=CONV>)', content)
        script_dict = {script_blocks[i]: script_blocks[i + 1] for i in range(1, len(script_blocks) - 1, 2)}
        script_pattern = f"script.{script_id_eval} "
        print('SCRIPT_PATTERN', script_pattern)
        for header, script_content in script_dict.items():
            if script_pattern in header:
                print('HEADER:', header)
                print('CONTENT:', script_content)
                make_evaluating_conversation(script_content,
                                             conversation, entity_properties, model, encoder, decoder, topk,
                                             log_file = False, print_statement = False)
    if script_id_eval == 0: # If evaluating withing the all training data
        make_evaluating_conversation(content,
                                     conversation, entity_properties, model, encoder, decoder, topk,
                                     log_file = False, print_statement = False)

    # Create the matrix from the responses the model gives (evaluation)
    # If the script id is there it creates an evaluation matrix based on the situation. If it is not there retrieves the situation
    # Optimal matrix is false since retrieving from conversation with Botchen
    # Entity properties is dynamical since we're updating it
    eval_matrix, eval_df = create_matrices(script_id_eval,
                                           0, # non-optimal matrix
                                           entity_properties,
                                           0) # don't store the space
    # Create the matrix from the training data (optimal)
    optimal_matrix, optimal_df = create_matrices(script_id_optimal,
                                                 1,
                                                 entity_properties, 0)

    matching_rows = eval_df.index.intersection(optimal_df.index)
    matching_columns = eval_df.columns.intersection(optimal_df.columns)
    non_matching_rows = eval_df.index.difference(optimal_df.index)
    non_matching_columns = eval_df.columns.difference(optimal_df.columns)

    if evaluating_framework == 1:  # Fill up the dimensionalities

        filtered_eval_df = eval_df.loc[eval_df.index.intersection(optimal_df.index), eval_df.columns.intersection(optimal_df.columns)]
        aligned_evaluation_df = filtered_eval_df.reindex(index=optimal_df.index, columns=optimal_df.columns, fill_value=0)
        print('ALIGNED EVALUATION', aligned_evaluation_df)

        # Compute the cosine similarity between rows
        similarity_df = pd.DataFrame(
            cosine_similarity(optimal_df.to_numpy(), aligned_evaluation_df.to_numpy()),
            index=optimal_df.index, columns=aligned_evaluation_df.index
        )
        print('ROW-WISE COSINE SIMILARITIES:\n', similarity_df)
        print('Average similarity rows', similarity_df.values.mean())

        # Compute cosine similarity between columns
        column_similarity_df = pd.DataFrame(
            cosine_similarity(optimal_df.T.to_numpy(), aligned_evaluation_df.T.to_numpy()),
            index=optimal_df.columns, columns=aligned_evaluation_df.columns
        )
        print('COLUMN-WISE COSINE SIMILARITIES:\n', column_similarity_df)
        print('AVERAGE COLUMN SIMILARITY:', column_similarity_df.values.mean())

    elif evaluating_framework == 2:  # Only match dimensions

        new_optimal_df = optimal_df.loc[matching_rows, matching_columns]
        new_eval_df = eval_df.loc[matching_rows, matching_columns]

        row_similarity_df = pd.DataFrame(
            cosine_similarity(new_optimal_df.to_numpy(), new_eval_df.to_numpy()),
            index=new_optimal_df.index,
            columns=new_eval_df.index
        )
        print('ROW-WISE COSINE SIMILARITIES:\n', row_similarity_df)
        print('Average similarity (rows):', row_similarity_df.values.mean())

        column_similarity_df = pd.DataFrame(
            cosine_similarity(new_optimal_df.T.to_numpy(), new_eval_df.T.to_numpy()),
            index=new_optimal_df.columns,
            columns=new_eval_df.columns
        )
        print('COLUMN-WISE COSINE SIMILARITIES:\n', column_similarity_df)
        print('Average similarity (columns):', column_similarity_df.values.mean())

    if evaluating_framework == 2:

        if not non_matching_rows.empty:
            print(f"Warning: The following rows in eval_df are not in optimal_df and will be ignored: {non_matching_rows.tolist()}")
        if not non_matching_columns.empty:
            print(f"Warning: The following columns in eval_df are not in optimal_df and will be ignored: {non_matching_columns.tolist()}")

        matching_rows = eval_df.index.intersection(optimal_df.index)
        matching_columns = eval_df.columns.intersection(optimal_df.columns)
        non_matching_columns = eval_df.columns.difference(optimal_df.columns)
        new_optimal_df = optimal_df.loc[matching_rows, matching_columns]
        new_eval_df = eval_df.loc[matching_rows, matching_columns]  # Also subset rows

        if not non_matching_columns.empty:
            print("Warning: The following columns in eval_df are not in optimal_df and will be ignored:")
            print(non_matching_columns.tolist())

        row_similarity_matrix = cosine_similarity(new_optimal_df.to_numpy(), new_eval_df.to_numpy())
        row_similarity_df = pd.DataFrame(row_similarity_matrix, index=new_optimal_df.index, columns=new_eval_df.index)
        print('ROW-WISE COSINE SIMILARITIES:\n', row_similarity_df)
        average_similarity_rows = np.mean(row_similarity_matrix)
        print('Average similarity (rows):', average_similarity_rows)

        column_similarity_matrix = cosine_similarity(new_optimal_df.T.to_numpy(), new_eval_df.T.to_numpy())
        column_similarity_df = pd.DataFrame(column_similarity_matrix, index=new_optimal_df.columns, columns=new_eval_df.columns)
        print('COLUMN-WISE COSINE SIMILARITIES:\n', column_similarity_df)
        average_column_similarity = np.mean(column_similarity_matrix)
        print('Average similarity (columns):', average_column_similarity)

    elif evaluating_framework == 3:  # RSA-based evaluation

        if not non_matching_rows.empty:
            print(f"Warning: The following rows in eval_df are not in optimal_df and will be ignored: {non_matching_rows.tolist()}")
        if not non_matching_columns.empty:
            print(f"Warning: The following columns in eval_df are not in optimal_df and will be ignored: {non_matching_columns.tolist()}")

        aligned_evaluation_df = eval_df.loc[matching_rows, matching_columns].reindex(index=optimal_df.index, columns=optimal_df.columns, fill_value=np.nan)
        print('ALIGNED EVAL 1', aligned_evaluation_df)

        # Fill NaN values with the mean of other columns in the row
        aligned_evaluation_df = aligned_evaluation_df.apply(lambda col: col.fillna(col.mean()), axis=0)
        print('ALIGNED EVAL 2', aligned_evaluation_df)

        # Fill NaN entire columns with the mean of the other values in the df
        for column in aligned_evaluation_df.columns:
            aligned_evaluation_df[column] = aligned_evaluation_df.apply(
                lambda row: row.drop(column).mean() if pd.isna(row[column]) else row[column],
                axis=1
            )
        print('ALIGNED EVAL 3', aligned_evaluation_df)

        if aligned_evaluation_df.isna().sum().sum() > 0:
            print("There are still NaN values in the aligned evaluation DataFrame.")
        if optimal_df.isna().sum().sum() > 0:
            print("There are still NaN values in the optimal DataFrame.")

        print('ENSURING', optimal_df.dtypes)  # Ensure all columns are numeric
        print('ENSURING', aligned_evaluation_df.dtypes)

        # Compute RSA Score
        optimal_dissimilarity = squareform(pdist(optimal_df, metric='cosine'))
        evaluation_dissimilarity = squareform(pdist(aligned_evaluation_df, metric='cosine'))
        rsa_score, _ = spearmanr(optimal_dissimilarity.flatten(), evaluation_dissimilarity.flatten())

        print("RSA Score:", rsa_score)