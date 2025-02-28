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
from utils import extract_entities_properties, make_evaluating_conversation
import argparse
import pandas as pd
import numpy as np
import re
import os

####################

directory = './data/chat/en/'
saving_directory = './data/chat/en/vectorial_space.csv'

@training.cli.command('create_matrices')
@click.argument('script_id', type = int)
@click.argument('optimal_matrix', type=int)
@click.argument('entity_properties_dynamical',required=False)
@click.argument('store_space', type=int, required=False)

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

########################################

@botchen_vectorial_space.cli.command('evaluation_with_vectorial_space')
@click.argument('module')
@click.argument('language')
@click.argument('topk')
# With this script I am creating a vectorial space from the responses the model gives to prompts from the training data in order to be abl>
# If evaluating on a situational script, see if to point the evaluation script on one situation and and if also to use the situation for the optimal one
def evaluation_with_vectorial_space(module, language, topk,
                                    evaluating_framework,
                                    script_id_eval = None,
                                    script_id_optimal = None,
                                    store_conversation = False):

    # Which model are we using?
    gpt_models = load_gpt_models(module, modules_path, language)
    model = gpt_models['chat'][0]
    encoder = gpt_models['chat'][1]
    decoder = gpt_models['chat'][3]
    print(f">> Loaded models {gpt_models.keys()}\n\n")

    if store_conversation:
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

    if script_id_eval: # If evaluating on only one script situation
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
    else: # If evaluating withing the all training data
        make_evaluating_conversation(content,
                                     conversation, entity_properties, model, encoder, decoder, topk,
                                     log_file = False, print_statement = False)

    # Create the matrix from the responses the model gives (evaluation)
    # If the script id is there it creates an evaluation matrix based on the situation. If it is not there retrieves the situation
    # Optimal matrix is false since retrieving from conversation with Botchen
    # Entity properties is dynamical since we're updating it
    eval_matrix, eval_df = create_matrices(script_id = script_id_eval,
                                           optimal_matrix = False,
                                           entity_properties_dynamical = entity_properties,
                                           store_space = False)
    # Create the matrix from the training data (optimal)
    optimal_matrix, optimal_df = create_matrices(script_id = script_id_optimal,
                                                 optimal_matrix = True,
                                                 store_space = False)

    if evaluating_framework == 1: # compute similarity looking at the columns (dimensions, sets, properties)
        # Align the new one with the index of the optimal matrix, filling the columns which are empty with 0 values
        aligned_evaluation_df = eval_df.reindex(index=optimal_df.index, fill_value=0)
        aligned_evaluation_df = aligned_evaluation_df.reindex(columns=optimal_df.columns, fill_value=0)
        print('ALIGNED EVALUATION', aligned_evaluation_df)

        # Compute the cosine similarity between rows
        similarity_df = cosine_similarity(optimal_df.to_numpy(), aligned_evaluation_df.to_numpy())
        similarity_df = pd.DataFrame(similarity_df, index=optimal_df.index, columns=aligned_evaluation_df.index)
        print('ROW-WISE COSINE SIMILARITIES:')
        print(similarity_df)
        average_similarity_rows = np.mean(similarity_df)
        print('Average similarity rows', average_similarity_rows)

        # Compute the cosine similarity between columns (transposed)
        # similarity_matrix_cols = cosine_similarity(optimal_df.T.to_numpy(), aligned_evaluation_df.T.to_numpy())
        # similarity_df_cols = pd.DataFrame(similarity_matrix_cols, index=optimal_df.columns, columns=aligned_evaluation_df.columns)

        # average_similarity_cols = np.mean(similarity_df_cols)

        column_similarities = []
        for col in optimal_df.columns:
            optimal_col = optimal_df[col].values.reshape(-1, 1)  # Reshape to make it a 2D column vector
            eval_col = aligned_evaluation_df[col].values.reshape(-1, 1)  # Reshape similarly
            similarity = cosine_similarity(optimal_col.T, eval_col.T)[0][0]  # Cosine similarity between two columns
            column_similarities.append((col, similarity))
            print('CosSim for {col}', column_similarities)
        column_similarity_df = pd.DataFrame(column_similarities, columns=["Column", "Cosine Similarity"])
        print('COLUMN-WISE COSINE SIMILARITIES:')
        print(column_similarity_df)

        average_column_similarity = np.mean([sim[1] for sim in column_similarities])
        print('AVERAGE COLUMN SIMILARITY:', average_column_similarity)