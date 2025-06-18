# EVALUATION SCRIPT

################## VECTORIAL SPACES EVALUATION METHOD

# The .**`/scripts/evaluation.py`** script evaluates the conversation output by comparing it with the optimal training data.
# The script creates vectorial spaces for both the evaluation and training data, with entities as rows and properties of them as columns.
# It then compares the two spaces using cosine similarity. Two frameworks are used for comparison.
# The first fits the evaluating space to the optimal space by adding missing dimensions (filled with zeros).
# The second matches the intersecting dimensions between the two spaces and compares them.
# Cosine similarity is computed for both rows and columns.

################## BLEU ALGORITHM METHOD

import os
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.translate.bleu_score import sentence_bleu
from os.path import dirname, realpath, join

################## VECTORIAL SPACES EVALUATION METHOD

def optimal_extract_entities(utterances):
    entity_properties = {}
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

def evaluation_extract_entities(utterances, print_discarded = False):
    entity_properties = {}
    discarded_items = []
    for utt in utterances:

        match = re.search(r'<u speaker=(?:HUM|BOT)>\s*\((.*?)\)', utt)
        if match:
            utt = match.group(1).strip()
        if (utt.startswith('(') and utt.endswith(')')):
            utt = utt.strip('()').strip()
        # For the surface to logic, we could also consider adding this part of the code
        # Now, since the conversation wasn't so good, I did not put it 
        # for utterance in bot_utterances:
        #         items = re.findall(r'\((.*?)\)', utterance)
        #         utterances.extend(items)

        utt = utt.strip()
        tokens = utt.split()

        if not tokens:
            discarded_items.append(f"Empty utterance after stripping: {utt}")
            continue

        entity_token = tokens[0]
        if not entity_token.endswith('.n'):
            discarded_items.append(f"Entity does not end with .n: {entity_token} in utterance: {utt}")
            continue

        entity = entity_token[:-2]
        seen_props = set()
        valid_props = []
        properties = []

        for prop in tokens[1:]:
            if prop.endswith('.n'):
                discarded_items.append(f"Discarded property (ends with .n): {prop} in utterance {utt}")
                continue
            if re.search(r'\.\w+', prop):
                discarded_items.append(f"Discarded property (contains suffix like .something): {prop}")
                continue
            if prop in seen_props:
                # discarded_items.append(f"Duplicate property in same utterance: {prop}")
                continue

            seen_props.add(prop)
            valid_props.append(prop)

        if entity not in entity_properties:
            entity_properties[entity] = set()
        entity_properties[entity].update(valid_props)

    if print_discarded:
        print("\n[Discarded Items]")
        for item in discarded_items:
            print("-", item)
    return entity_properties

def read_file(file_path):
    """Reads content from a single file."""
    with open(file_path, 'r') as file:
        return file.read()

def read_evaluation_files(directory):
    """Reads content from all evaluation files in the directory."""
    content = ""
    for file_name in os.listdir(directory):
        if file_name.startswith('evaluation_situation') and file_name.endswith('.txt'):
            file_path = os.path.join(directory, file_name)
            content += read_file(file_path)
    return content

def save_matrix_to_csv(df, saving_directory, evaluating_framework, optimal_script):
    """Saves the DataFrame to CSV in the given directory."""
    if not os.path.exists(saving_directory):
        os.makedirs(saving_directory)
        print(f"Directory {saving_directory} created.")
    
    file_name = f"optimal_df_{evaluating_framework}.csv" if optimal_script else f"eval_df_{evaluating_framework}.csv"
    file_path = os.path.join(saving_directory, file_name)
    df.to_csv(file_path, index=True)
    print(f"Matrix saved to {file_path}")

def retraction_evaluation(eval_df, optimal_df, matching_rows, matching_columns):
    """Handles retraction evaluation logic."""
    combined_indices = eval_df.index.union(optimal_df.index)
    combined_columns = eval_df.columns.union(optimal_df.columns)
    new_eval_df = pd.DataFrame(0, index=combined_indices, columns=combined_columns)
    new_optimal_df = pd.DataFrame(0, index=combined_indices, columns=combined_columns)
    new_eval_df.loc[matching_rows, matching_columns] = eval_df.loc[matching_rows, matching_columns]
    new_optimal_df.loc[matching_rows, matching_columns] = optimal_df.loc[matching_rows, matching_columns]
    return new_eval_df, new_optimal_df

def calculate_similarities(optimal_df, eval_df):
    """Calculates cosine similarities between rows and columns."""
    row_similarity = pd.Series(cosine_similarity(optimal_df.to_numpy(), eval_df.to_numpy()).diagonal(), index=optimal_df.index)
    column_similarity = pd.Series(cosine_similarity(optimal_df.T.to_numpy(), eval_df.T.to_numpy()).diagonal(), index=optimal_df.columns)
    return row_similarity, column_similarity

def print_similarity_results(row_similarity, column_similarity, evaluation_type):
    """Prints similarity results to the terminal."""
    print(f"    {evaluation_type}")
    print(f"    Average Row Similarity: {row_similarity.mean():.3f}")
    # print(f"Top 10 Most Similar Rows:")
    # print(row_similarity.nlargest(10))
    
    print(f"    Average Column Similarity: {column_similarity.mean():.3f}")
    # print(f"Top 10 Most Similar Columns:")
    # print(column_similarity.nlargest(10))

def extract_entities_properties(content, optimal_script, format_type = None):
    if optimal_script:
        if format_type == 'logic_to_logic':
            utterance_pattern = re.compile(r'<u speaker=[^>]*>\((.*?)\)</u>')
            utterances = utterance_pattern.findall(content)
            entity_properties = optimal_extract_entities(utterances)
        if format_type == 'surface_to_logic':
            bot_utterances = re.findall(r'<u speaker=BOT>(.*?)</u>', content)
            utterances = []
            for utterance in bot_utterances:
                items = re.findall(r'\((.*?)\)', utterance)
                utterances.extend(items)
            entity_properties = optimal_extract_entities(utterances)
    else:
        utterance_pattern = re.compile(r">> Response: (.*?)\n\n")
        utterances = utterance_pattern.findall(content)
        entity_properties = evaluation_extract_entities(utterances,print_discarded = False)

    return entity_properties

def create_matrices(evaluating_framework, format_type, evaluation_file_path, optimal_script=False, saving_directory=None):
    """
    Creates an activation matrix from a script, either an optimal or evaluation script.
    Optionally saves the matrix and returns entity properties.
    
    Parameters:
    - optimal_script (bool): Use optimal script (True) or evaluation data (False).
    - saving_directory (str or None): Directory to save the matrix CSV. If None, no saving happens.
    
    Returns:
    - activation_matrix (list): Activation matrix.
    """
    content = ""
    
    # Load the content from the script(s)
    if optimal_script:
        if format_type == 'logic_to_logic':
            content = read_file(
                os.path.join(parent_dir, "data", "training", "prompt_files", f"prompt_{format_type}.txt"))

        if format_type == 'surface_to_logic':
            content = read_file(
                os.path.join(parent_dir, "data", "training", "permuted_files", f"permuted_{format_type}.txt"))
        entity_properties = extract_entities_properties(content, optimal_script=True, format_type = format_type)
    else:
        content = read_evaluation_files(evaluation_file_path)
        entity_properties = extract_entities_properties(content, optimal_script=False)
    
    # Create sorted list of unique properties and entities
    all_properties = sorted(set(prop for props in entity_properties.values() for prop in props))
    entities = list(entity_properties.keys())
    
    # Generate the activation matrix
    activation_matrix = [[1 if prop in entity_properties[entity] else 0 for prop in all_properties] for entity in entities]
    
    # Convert to DataFrame
    df = pd.DataFrame(activation_matrix, columns=all_properties, index=entities)
    
    # Save to CSV if saving directory is provided
    if saving_directory:
        save_matrix_to_csv(df, saving_directory, evaluating_framework, optimal_script)
    
    return activation_matrix, df

def evaluation_with_vectorial_space(evaluating_framework, format_type):
    """
    Compares vectorial spaces from an evaluation script and an optimal training script.
    
    Parameters:
    - evaluating_framework (int): 1 for Fill-up, 2 for Retraction.
    
    Returns:
    - None: Prints evaluation results to the terminal.
    """
    # Print evaluation start
    print(f"\nEvaluating with framework {evaluating_framework}")
    
    # Create the evaluation and optimal matrices
    eval_matrix, eval_df = create_matrices(evaluating_framework, 
        format_type,
        os.path.join(parent_dir, "data", "evaluation_data", f"evaluation_{format_type}"),
        saving_directory = os.path.join(parent_dir, "data", "vectorial_spaces", f"evaluation_{format_type}"))

    optimal_matrix, optimal_df = create_matrices(evaluating_framework, 
        format_type,
        os.path.join(parent_dir, "data", "evaluation_data", f"evaluation_{format_type}"),
        saving_directory = os.path.join(parent_dir, "data", "vectorial_spaces", f"evaluation_{format_type}"), 
        optimal_script=True)
    
    # Get common rows and columns
    matching_rows = eval_df.index.intersection(optimal_df.index)
    matching_columns = eval_df.columns.intersection(optimal_df.columns)
    
    # Perform evaluation based on the framework
    if evaluating_framework == 1:
        new_eval_df = eval_df.loc[matching_rows, matching_columns].reindex(index=optimal_df.index, columns=optimal_df.columns, fill_value=0)
        new_optimal_df = optimal_df
        evaluation_type = "Fill-up dimensionalities evaluation"
    elif evaluating_framework == 2:
        new_eval_df, new_optimal_df = retraction_evaluation(eval_df, optimal_df, matching_rows, matching_columns)
        evaluation_type = "Retraction evaluation"
    
    # Calculate and print similarities
    row_similarity, column_similarity = calculate_similarities(new_optimal_df, new_eval_df)
    print_similarity_results(row_similarity, column_similarity, evaluation_type)

################ BLEU EVALUATION

def parse_references(file_path, format_type):
    with open(file_path, 'r') as f:
        content = f.read()

    if format_type == 'logic_to_surface':
        matches = re.findall(
            r'<a script\.(\d+) type=DSC>\s*<u speaker=HUM>(.*?)</u>\s*<u speaker=BOT>(.*?)</u>\s*</a>', 
            content, re.DOTALL
        )
        ref_dict = {}
        for _, hum_text, bot_text in matches:
            ref_dict.setdefault(hum_text, set()).add(bot_text)
        return ref_dict
    else:  # surface_to_surface or sandwich
        matches = re.findall(r'<u speaker=[^>]*>(.*?)</u>', content, re.DOTALL)
        return [m.split() for m in matches]

def parse_candidates(file_path, format_type):
    with open(file_path, 'r') as f:
        content = f.read()

    if format_type == 'logic_to_surface':
        matches = re.findall(r">> Prompt: (.*?)\s+>> Response: (.*?)\n\n", content, re.DOTALL)
        cand_dict = {}
        for prompt, response in matches:
            cand_dict.setdefault(prompt, []).append(response)
        return cand_dict

    else:
        matches = re.findall(r">> Prompt: (.*?)\s+>> Response: (.*?)\n\n", content, re.DOTALL)
        return [response.split() for _, response in matches]

def compute_bleu(tokenized_refs, tokenized_cands, n_gram, format_type):
    weights = tuple(1.0 if i == 0 else 0.0 for i in range(n_gram))
    bleu_scores = []

    if format_type == 'logic_to_surface':
        for prompt in tokenized_cands:
            if prompt in tokenized_refs:
                references = [r.split() for r in tokenized_refs[prompt]]
                candidates = [c.split() for c in tokenized_cands[prompt]]
                scores = [sentence_bleu(references, cand, weights=weights) for cand in candidates]
                bleu_scores.append(max(scores))
    else:
        for cand in tokenized_cands:
            if format_type == 'sandwich' and any(x.startswith('<u') or '(' in x or ')' in x or 'speaker=' in x for x in cand):
                bleu_scores.append(0)
                continue
            scores = [sentence_bleu([ref], cand, weights=weights) for ref in tokenized_refs]
            bleu_scores.append(max(scores))

    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

def bleu_algorithm_scoring(file_path_references, file_path_candidates, n_gram, format_type):
    references = parse_references(file_path_references, format_type)
    candidates = parse_candidates(file_path_candidates, format_type)
    return compute_bleu(references, candidates, n_gram, format_type)

########## LOGIC TO LOGIC

logging.info('\nVectorial Space Evaluation - Logic to Logic Data')
evaluation_with_vectorial_space(1, 'logic_to_logic')
evaluation_with_vectorial_space(2, 'logic_to_logic')

#################

############ SURFACE TO LOGIC

logging.info('\nVectorial Space Evaluation - Surface to Logic Data')
evaluation_with_vectorial_space(1, 'surface_to_logic')
evaluation_with_vectorial_space(2, 'surface_to_logic')

########### BLEU

############## LOGICAL TO SURFACE

logging.info('\nBLEU Algorithm Evaluation - Logic to Surface Data')
for n_gram_value in range(1,3):
    score_logic_surface = bleu_algorithm_scoring(
        os.path.join(parent_dir, "data", "training", "permuted_files", "permuted_logic_to_surface.txt"),
        os.path.join(parent_dir, "data", "evaluation_data", "evaluation_logic_to_surface", "evaluation_situation0.txt"),
        n_gram_value, 'logic_to_surface')

    logging.info(f'    Average BLEU score, n-grams {n_gram_value} is {score_logic_surface}')

########### SURFACE

logging.info('\nBLEU Algorithm Evaluation - Surface Data')
for n_gram_value in range(1,3):
    score_surface = bleu_algorithm_scoring(
        os.path.join(parent_dir, "data", "training", "prompt_files", "prompt_surface_to_surface.txt"),
        os.path.join(parent_dir, "data", "evaluation_data", "evaluation_surface", "evaluation_situation0.txt"),
        n_gram_value, 'surface_to_surface')

    logging.info(f'    Average BLEU score, n-grams {n_gram_value} is {score_surface}')

################ SANDWICH

logging.info('\nBLEU Algorithm Evaluation - Sandwich Data')
for n_gram_value in range(1,3):
    score_sandwich = bleu_algorithm_scoring(
        os.path.join(parent_dir, "data", "training", "prompt_files", "prompt_sandwich.txt"),
        os.path.join(parent_dir, "data", "evaluation_data", "emma_sandwich", "evaluation_situation0.txt"),
        n_gram_value, 'sandwich')
    logging.info(f'    Average BLEU score, n-grams {n_gram_value} is {score_sandwich}')

