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

# Create vectorial spaces from different content.
# script_id: indicates the specific situation we want to build the space on. If it is 0 it retrieves the all data.
# optimal_script: if it's retrieving the situation from the training data or from Botchen conversation.
# saving_directory: if we want to store the space, indicate the directory
def create_matrices(script_id, optimal_script=False, saving_directory=None):
    content = ''
    if optimal_script:
        if script_id == 0:
            for file_name in os.listdir('./data/training_data/'):
                if file_name.startswith('ideallanguage') and file_name.endswith('.txt'):
                    file_path = os.path.join('./data/training_data/', file_name)
                    with open(file_path, 'r') as file:
                        script_content = file.read()
                        content += script_content
        elif script_id != 0:
            file_path = f'./data/training_data/ideallanguage_{script_id}.txt'
            with open(file_path, 'r') as file:
                content = file.read()
        entity_properties = extract_entities_properties(content, optimal_script=True)
    else:
        file_path = f'./data/evaluation_data/evaluation_situation{script_id}.txt'
        with open(file_path, 'r') as file:
            content = file.read()
        entity_properties = extract_entities_properties(content, optimal_script=False)

    all_properties = sorted(set(prop for props in entity_properties.values() for prop in props))
    entities = list(entity_properties.keys())
    activation_matrix = []
    for entity in entities:
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
                                    script_id_optimal):

    print(f"Evaluating with framework: {evaluating_framework}, script eval ID: {script_id_eval}, script optimal ID: {script_id_optimal}")

    eval_matrix, eval_df = create_matrices(script_id_eval, saving_directory='./data/vectorial_spaces/evaluation/')

    optimal_matrix, optimal_df = create_matrices(script_id_optimal, optimal_script=True, saving_directory='./data/vectorial_spaces/optimal/')

    matching_rows = eval_df.index.intersection(optimal_df.index)
    matching_columns = eval_df.columns.intersection(optimal_df.columns)

    if evaluating_framework == 1:
        # Clean the evaluated space from dimensions which are not in the optimal space
        # If there are dimensions in the optimal space which are not in the evaluated space fill the ones in the evaluated space with zero
        print('EVALUATION 1 - Fill up dimensionalities evaluation')
        new_eval_df = eval_df.loc[eval_df.index.intersection(optimal_df.index), eval_df.columns.intersection(optimal_df.columns)]
        new_eval_df = new_eval_df.reindex(index=optimal_df.index, columns=optimal_df.columns, fill_value=0)
        new_optimal_df = optimal_df

    elif evaluating_framework == 2:
        # Extract from both the training and the evaluated file just the matching dimensions (intersections)
        print('EVALUATION 2 - Retraction evaluation')
        new_optimal_df = optimal_df.loc[matching_rows, matching_columns]
        new_eval_df = eval_df.loc[matching_rows, matching_columns]
    else:
        print('WARNING - no evaluation framework selected')

    row_similarity_df = pd.DataFrame(
        cosine_similarity(new_optimal_df.to_numpy(), new_eval_df.to_numpy()),
        index=new_optimal_df.index, columns=new_eval_df.index)

    row_diagonal_values = row_similarity_df.to_numpy().diagonal()
    row_diagonal_series = pd.Series(row_diagonal_values, index=row_similarity_df.index)

    print('Average similarity rows', "{:.3f}".format(row_diagonal_series.values.mean()))
    row_10_highest = row_diagonal_series.nlargest(10)
    print('Most similar rows:\n', row_10_highest)
    #row_10_lowest = rows_similarities.nsmallest(10)

    column_similarity_df = pd.DataFrame(
        cosine_similarity(new_optimal_df.T.to_numpy(), new_eval_df.T.to_numpy()),
        index=new_optimal_df.columns, columns=new_eval_df.columns)
    col_diagonal_values = column_similarity_df.to_numpy().diagonal()
    col_diagonal_series = pd.Series(col_diagonal_values, index=column_similarity_df.index)

    print('Average Columns Similarity:', "{:.3f}".format(col_diagonal_series.values.mean()))
    col_10_highest = col_diagonal_series.nlargest(10)
    print('Most similar columns:\n', col_10_highest)
    # col_10_lowest = column_similarities.nsmallest(10)

function_map = {
    "evaluation": evaluation_with_vectorial_space
}

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("evaluation", type=str, help="Function to run the evaluation")
    parser.add_argument("evaluating_framework", type=int, help="1 or 2  based on the type of evaluation you want to make")
    parser.add_argument("script_id_eval", type=int, help="Situation to evaluate from Botchen conversation (0 means all)")
    parser.add_argument("script_id_optimal", type=int, help="Situation to evaluate with from training data, optimal (0 means all)")

    args = parser.parse_args()
    func = function_map.get(args.evaluation)
    if func:
        func(args.evaluating_framework, args.script_id_eval, args.script_id_optimal)
    else:
        print(f"Function '{args.evaluation}' not found.")

if __name__ == "__main__":
    main()
