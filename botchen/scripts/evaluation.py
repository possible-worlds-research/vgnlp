# The .**`/scripts/evaluation.py`** script evaluates the conversation output by comparing it with the optimal training data.
# The script creates vectorial spaces for both the evaluation and training data, with entities as rows and properties of them as columns.
# It then compares the two spaces using cosine similarity. Two frameworks are used for comparison.
# The first fits the evaluating space to the optimal space by adding missing dimensions (filled with zeros).
# The second matches the intersecting dimensions between the two spaces and compares them.
# Cosine similarity is computed for both rows and columns.

# **Usage:** (A) Evaluating framework: ```python3 ./scripts/evaluation.py evaluation 1 2 2```.
# Evaluates within first evaluation framework, evaluating situation 2, optimal situation 2.
#
# (B) Create matrices: ```python ./scripts/evaluation.py create_matrices 1 --optimal_script --saving_directory './data/vectorial_spaces/optimal/'```.
# Evaluating  with situation 1, from training (otpimal) data and saving the space in the directory.
# If one wants to make them from evaluating data: ```python3 ./scripts/evaluation.py create_matrices 2 --saving_directory './data/vectorial_spaces/evaluation/'```.
# In general, the user will need **`./data/evaluation_data/*'** as input
# and generates './data/vectorial_spaces/evaluation/*' or './data/vectorial_spaces/optimal/*' as csv outputs.
# The user can dinamically select the situation in which to test the model on and if to store it.

import os
from os.path import dirname, realpath, join
from flask import Blueprint
import click
import torch
from datetime import datetime
import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import argparse

# Extract entities and properties from content
def extract_entities_properties(content, optimal_script):
    entity_properties = {}
    if optimal_script:
        utterance_pattern = re.compile(r'<u speaker=[^>]*>\((.*?)\)</u>')
    else:
        utterance_pattern = re.compile(r'<u speaker=[^>]+>\((.*?)\)', re.DOTALL)
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
    return entity_properties

def create_matrices(script_id, optimal_script=False, saving_directory=None):
    """
    This function creates an activation matrix based on entity properties from a script.
    It either processes a specific script or an optimal script and generates a matrix of
    entity-property activations. The matrix is optionally saved to a directory if specified.

    Parameters:
    - script_id (int): The ID of the script to evaluate.
    - optimal_script (bool): Whether to use an optimal script (True) or evaluation data (False).
    - saving_directory (str or None): Directory to save the resulting matrix as a CSV file. If None, the matrix is not saved.

    Returns:
    - activation_matrix (list): A list of lists representing the activation matrix.
    - df (pandas.DataFrame): The activation matrix in a pandas DataFrame format.
    """
    content = ''
    if optimal_script:
        # If script_id is 0, process all 'ideallanguage' files in the directory
        if script_id == 0:
            # Look for files that start with 'ideallanguage' and end with '.txt'
            for file_name in os.listdir('./data/training_data/'):
                if file_name.startswith('ideallanguage') and file_name.endswith('.txt'):
                    file_path = os.path.join('./data/training_data/', file_name)
                    with open(file_path, 'r') as file:
                        script_content = file.read()
                        content += script_content
        elif script_id != 0:
            # If script_id is not 0, load the specific optimal script for that ID
            file_path = f'./data/training_data/ideallanguage_{script_id}.txt'
            with open(file_path, 'r') as file:
                content = file.read()
        entity_properties = extract_entities_properties(content, optimal_script=True)
    # If not an optimal script, load the evaluation data for the given script_id
    else:
        if script_id == 0:
            for file_name in os.listdir('./data/evaluation_data/'):
                if file_name.startswith('evaluation_situation') and file_name.endswith('.txt'):
                    file_path = os.path.join('./data/evaluation_data/', file_name)
                    with open(file_path, 'r') as file:
                        script_content = file.read()
                        content += script_content
        elif script_id != 0:
            file_path = f'./data/evaluation_data/evaluation_situation{script_id}.txt'
            with open(file_path, 'r') as file:
                content = file.read()
        entity_properties = extract_entities_properties(content, optimal_script=False)

    # Create a sorted list of all unique properties across entities
    all_properties = sorted(set(prop for props in entity_properties.values() for prop in props))
    # Create a list of entities (keys from the entity_properties dictionary)
    entities = list(entity_properties.keys())
    activation_matrix = []
    # For each entity, create a row in the activation matrix
    for entity in entities:
        # A row is a list of 1s and 0s, where 1 indicates that the entity has a particular property
        row = [1 if prop in entity_properties[entity] else 0 for prop in all_properties]
        activation_matrix.append(row)
    df = pd.DataFrame(activation_matrix, columns=all_properties, index=entities)
    if saving_directory:
        if not os.path.exists(saving_directory):
            os.makedirs(saving_directory)
            print(f"Directory {saving_directory} created.")
        if optimal_script:
            file_path = os.path.join(saving_directory, f"optimal_df_{script_id}.csv")
        else:
            file_path = os.path.join(saving_directory, f"eval_df_{script_id}.csv")
        df.to_csv(file_path, index=True)
        print(f'Matrix has been stored as {file_path}')
    else:
        print('No saving directory provided, matrix will not be saved.')
    return activation_matrix, df

# Evaluate the similarity between the spaces, created from optimal training dat and Botchen conversation
# evaluating framework: 1 Fill up dimensionalities evaluation; 2 Retraction evaluation
# script_id_eval and script_id_optimal, which situations from which to retrieve from
def evaluation_with_vectorial_space(evaluating_framework,
                                    script_id_eval,
                                    script_id_optimal, log_file=None):
    """
    Evaluates the similarity between vector spaces derived from an evaluation script
    and an optimal training script. The function supports two evaluation frameworks:

    1. Fill-up Dimensionalities Evaluation: Ensures the evaluated space contains all
       dimensions from the optimal space, filling in missing values with zero.
    2. Retraction Evaluation: Compares only the shared dimensions between the evaluated
       and optimal spaces.

    Parameters:
    - evaluating_framework (int): The evaluation approach to use (1 or 2).
    - script_id_eval (int): The ID of the evaluation script to process.
    - script_id_optimal (int): The ID of the optimal script (training reference).

    Returns:
    - None: Prints evaluation results including average similarity scores
      and the top 10 most similar rows and columns.
    """
    print(f"Evaluating with framework: {evaluating_framework}, script eval ID: {script_id_eval}, script optimal ID: {script_id_optimal}")
    if log_file:
        with open(log_file, 'a') as file:
            file.write(f"\nNEWNEW Evaluating with framework: {evaluating_framework}, script eval ID: {script_id_eval}, script optimal ID: {script_id_optimal}\n")

    # Generate evaluation vector space (matrix & DataFrame)
    eval_matrix, eval_df = create_matrices(script_id_eval, saving_directory='./data/vectorial_spaces/evaluation/')
    # Generate optimal vector space (matrix & DataFrame)
    optimal_matrix, optimal_df = create_matrices(script_id_optimal, optimal_script=True, saving_directory='./data/vectorial_spaces/optimal/')

    # Identify matching rows (entities) and columns (features) between the evaluation and optimal datasets
    matching_rows = eval_df.index.intersection(optimal_df.index)
    matching_columns = eval_df.columns.intersection(optimal_df.columns)

    if evaluating_framework == 1:
        print('EVALUATION 1 - Fill up dimensionalities evaluation')
        # Retain only dimensions present in both datasets
        new_eval_df = eval_df.loc[eval_df.index.intersection(optimal_df.index), eval_df.columns.intersection(optimal_df.columns)]
        # Ensure the evaluated space includes all optimal dimensions, filling missing values with zero
        new_eval_df = new_eval_df.reindex(index=optimal_df.index, columns=optimal_df.columns, fill_value=0)
        new_optimal_df = optimal_df
        if log_file:
            with open(log_file, 'a') as file:
                file.write(f"\nEVALUATION 1 - Fill up dimensionalities evaluation\n")

    elif evaluating_framework == 2:
        print('EVALUATION 2 - Retraction evaluation')
        # Compare only shared dimensions from both datasets
        new_optimal_df = optimal_df.loc[matching_rows, matching_columns]
        new_eval_df = eval_df.loc[matching_rows, matching_columns]
        if new_optimal_df.empty or new_eval_df.empty:
            print("None - No intersection")
            if log_file:
                with open(log_file, 'a') as file:
                    file.write("\nEVALUATION 2 - Retraction evaluation\nNone - No intersection\n")
            return
        else:
            if log_file:
                with open(log_file, 'a') as file:
                    file.write(f"\nEVALUATION 2 - Retraction evaluation\n")
    else:
        print('WARNING - no evaluation framework selected')

    # Compute cosine similarity between row vectors (entity-level similarity)
    row_similarity_df = pd.DataFrame(
        cosine_similarity(new_optimal_df.to_numpy(), new_eval_df.to_numpy()),
        index=new_optimal_df.index, columns=new_eval_df.index)

    # Extract diagonal values (self-similarities) from the row similarity matrix
    row_diagonal_series = pd.Series(row_similarity_df.to_numpy().diagonal(), index=row_similarity_df.index)

    print(f'Average Rows Similarity: {row_diagonal_series.mean():.3f}')
    print('Most similar rows:\n', row_diagonal_series.nlargest(10))
    if log_file:
        with open(log_file, 'a') as file:
            file.write(f"\n\nAverage Rows Similarity: {row_diagonal_series.mean():.3f}\n\n Most similar rows:\n{row_diagonal_series.nlargest(10)}\n")

    # Compute cosine similarity between column vectors (feature-level similarity)
    column_similarity_df = pd.DataFrame(
        cosine_similarity(new_optimal_df.T.to_numpy(), new_eval_df.T.to_numpy()),
        index=new_optimal_df.columns, columns=new_eval_df.columns)
    # Extract diagonal values (self-similarities) from the column similarity matrix
    col_diagonal_series = pd.Series(column_similarity_df.to_numpy().diagonal(), index=column_similarity_df.index)

    # Display feature-level similarity results
    print(f'Average Column Similarity: {col_diagonal_series.mean():.3f}')
    print('Most similar columns:\n', col_diagonal_series.nlargest(10))
    if log_file:
        with open(log_file, 'a') as file:
            file.write(f"\n\nAverage Column Similarity: {col_diagonal_series.mean():.3f}\n\nMost similar columns:\n{col_diagonal_series.nlargest(10)}\n")

# function_map = {
#     "evaluation": evaluation_with_vectorial_space,
#     "create_matrices": create_matrices
# }

# def main():
#     parser = argparse.ArgumentParser(description="Run evaluation or create activation matrices.")
#
#     parser.add_argument("function", type=str, choices=function_map.keys(),
#                         help="Function to execute: 'evaluation' or 'create_matrices'")
#
#     # Common arguments for both functions
#     parser.add_argument("script_id", type=int, help="Script ID to process (0 means all)")
#
#     # Arguments specific to evaluation
#     parser.add_argument("--evaluating_framework", type=int, choices=[1, 2],
#                         help="Evaluation framework: 1 (Fill up dimensionalities) or 2 (Retraction evaluation)")
#     parser.add_argument("--script_id_optimal", type=int,
#                         help="Optimal script ID for comparison (only for evaluation)")
#
#     # Arguments specific to create_matrices
#     parser.add_argument("--optimal_script", action="store_true",
#                         help="Flag to indicate processing an optimal script")
#     parser.add_argument("--saving_directory", type=str, default=None,
#                         help="Directory to save the activation matrix (optional)")
#
#     args = parser.parse_args()
#
#     func = function_map.get(args.function)
#
#     if func == evaluation_with_vectorial_space:
#         if args.evaluating_framework is None or args.script_id_optimal is None:
#             print("Error: 'evaluation' requires --evaluating_framework and --script_id_optimal.")
#             return
#         func(args.evaluating_framework, args.script_id, args.script_id_optimal)
#
#     elif func == create_matrices:
#         func(args.script_id, args.optimal_script, args.saving_directory)
#
#     else:
#         print(f"Function '{args.function}' not found.")
#
# if __name__ == "__main__":
#     main()

open('./data/evaluation_file.txt', 'w').close()  # This clears the file by opening & immediately closing it

for script_id_eval in range (0,13):
    for script_id_optimal in range(0,13):
        evaluation_with_vectorial_space(1, script_id_eval, script_id_optimal, log_file='./data/evaluation_file.txt')
