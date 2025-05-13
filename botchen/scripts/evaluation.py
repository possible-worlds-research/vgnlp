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
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.translate.bleu_score import sentence_bleu

################## VECTORIAL SPACES EVALUATION METHOD

# Extract entities and properties from content
def extract_entities_properties(content, optimal_script):
    entity_properties = {}
    if optimal_script:
        utterance_pattern = re.compile(r'<u speaker=[^>]*>\((.*?)\)</u>')
    else:
        utterance_pattern = re.compile(r'<u speaker=[^>]+>\(([^)>]+)\)', re.DOTALL)
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

def count_entities_properties(entity_properties):
    num_entities = len(entity_properties)  # Number of unique entities
    all_properties = set(prop for props in entity_properties.values() for prop in props)
    num_properties = len(all_properties)  # Unique property count
    return num_entities, num_properties

def create_matrices(script_id, evaluation_file_path, optimal_script=False, saving_directory=None, entity_check=False):
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
        with open('./data/training/prompt_logical.txt', 'r') as file:
            script_content = file.read()
            content += ''.join(script_content)
        if script_id != 0:
            # If script_id is not 0, extract the specific optimal script for that ID
            content = re.findall(fr"(<script\.{script_id} type=CONV>.*?</script\.{script_id}>)", content, re.DOTALL)
            content = ''.join(content)
        entity_properties = extract_entities_properties(content, optimal_script=True)
    # If not an optimal script, load the evaluation data for the given script_id
    else:
        if script_id == 0:
            for file_name in os.listdir(evaluation_file_path):
                if file_name.startswith('evaluation_situation') and file_name.endswith('.txt'):
                    file_path = os.path.join(evaluation_file_path, file_name)
                    with open(file_path, 'r') as file:
                        script_content = file.read()
                        content += ''.join(script_content)
                        # content += script_content
        elif script_id != 0:
            file_path = f'./data/evaluation_data/evaluation_logical/evaluation_situation{script_id}.txt'
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
        # print(f'Matrix has been stored as {file_path}')
    else:
        # print('No saving directory provided, matrix will not be saved.')
        None
    if entity_check:
        return entity_properties
    return activation_matrix, df

# Evaluate the similarity between the spaces, created from optimal training dat and Botchen conversation
# evaluating framework: 1 Fill up dimensionalities evaluation; 2 Retraction evaluation
# script_id_eval and script_id_optimal, which situations from which to retrieve from
def evaluation_with_vectorial_space(evaluating_framework,
                                    script_id_eval,
                                    script_id_optimal, log_file=None, optimal_to_optimal=False):
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
    - log_file (bool): If one wants to store the evaluation on a file
    - optimal_to_optimal (bool): True if comparing optimal spaces (retains optimal one)

    Returns:
    - None: Prints evaluation results including average similarity scores
      and the top 10 most similar rows and columns.
    """
    # print(f"Evaluating with framework: {evaluating_framework}, script eval ID: {script_id_eval}, script optimal ID: {script_id_optimal}")
    if log_file:
        with open(log_file, 'a') as file:
            file.write(f"\nNEWNEW Evaluating with framework: {evaluating_framework}, script eval ID: {script_id_eval}, script optimal ID: {script_id_optimal}\n")

    # Generate evaluation vector space (matrix & DataFrame)
    if optimal_to_optimal is True:
        eval_matrix, eval_df = create_matrices(script_id_eval, './data/evaluation_data/evaluation_logical/', optimal_script=True)
    else:
        eval_matrix, eval_df = create_matrices(script_id_eval, './data/evaluation_data/evaluation_logical/', saving_directory='./data/data_analysis/logical/vectorial_spaces/evaluation/')
    # Generate optimal vector space (matrix & DataFrame)
    optimal_matrix, optimal_df = create_matrices(script_id_optimal, './data/evaluation_data/evaluation_logical/', optimal_script=True, saving_directory='./data/data_analysis/logical/vectorial_spaces/optimal/')

    # Identify matching rows (entities) and columns (features) between the evaluation and optimal datasets
    matching_rows = eval_df.index.intersection(optimal_df.index)
    matching_columns = eval_df.columns.intersection(optimal_df.columns)

    if evaluating_framework == 1:
        # print('EVALUATION 1 - Fill up dimensionalities evaluation')
        # Retain only dimensions present in both datasets
        new_eval_df = eval_df.loc[matching_rows, matching_columns]
        # Ensure the evaluated space includes all optimal dimensions, filling missing values with zero
        new_eval_df = new_eval_df.reindex(index=optimal_df.index, columns=optimal_df.columns, fill_value=0)
        new_optimal_df = optimal_df
        if log_file:
            with open(log_file, 'a') as file:
                file.write(f"\nEVALUATION 1 - Fill up dimensionalities evaluation\n")

    elif evaluating_framework == 2:
        # print('EVALUATION 2 - Retraction evaluation')
        combined_indices = eval_df.index.union(optimal_df.index)
        combined_columns = eval_df.columns.union(optimal_df.columns)
        new_eval_df = pd.DataFrame(0, index=combined_indices, columns=combined_columns)
        new_optimal_df = pd.DataFrame(0, index=combined_indices, columns=combined_columns)
        new_eval_df.loc[matching_rows, matching_columns] = eval_df.loc[matching_rows, matching_columns]
        new_optimal_df.loc[matching_rows, matching_columns] = optimal_df.loc[matching_rows, matching_columns]

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

    # print(f'Average Rows Similarity: {row_diagonal_series.mean():.3f}')
    # print('Most similar rows:\n', row_diagonal_series.nlargest(10))
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
    # print(f'Average Column Similarity: {col_diagonal_series.mean():.3f}')
    # print('Most similar columns:\n', col_diagonal_series.nlargest(10))
    if log_file:
        with open(log_file, 'a') as file:
            file.write(f"\n\nAverage Column Similarity: {col_diagonal_series.mean():.3f}\n\nMost similar columns:\n{col_diagonal_series.nlargest(10)}\n")

################ REPRESENTING GRAPHICALLY

def make_csv_from_data(file_path, name):
    with open(file_path, 'r') as file:
        data = file.read()
    evaluation_pattern = re.compile(r"Evaluating with framework: (\d+), script eval ID: (\d+), script optimal ID: (\d+)")
    row_similarity_pattern = re.compile(r"Average Rows Similarity:\s+([\d\.]+)")
    column_similarity_pattern = re.compile(r"Average Column Similarity:\s+([\d\.]+)")

    summary_data = []
    evaluations = data.strip().split("NEWNEW")

    for eval_block in evaluations[0:]:  # Skip empty first split
        eval_info = evaluation_pattern.search(eval_block)
        row_sim = row_similarity_pattern.search(eval_block)
        col_sim = column_similarity_pattern.search(eval_block)

        if eval_info and row_sim and col_sim:
            framework, eval_script, opt_script = eval_info.groups()
            avg_row_sim = float(row_sim.group(1))
            avg_col_sim = float(col_sim.group(1))

            summary_data.append([framework, eval_script, opt_script, avg_row_sim, avg_col_sim])

    summary_df = pd.DataFrame(summary_data, columns=["Framework", "Eval Script", "Optimal Script", "Avg Row Similarity", "Avg Column Similarity"])
    summary_df.to_csv(f"{name}", index=False)

def heatmap(file_path, row_column, eval_optimal, number_eval, log_file):
    # row_column= Row/Column based on what you want to evaluate
    # eval_optimal= Optimal/Eval based on what you are comparing
    # number_eval = 1/2 if evaluating with 1 or 2 eval
    df = pd.read_csv(file_path)
    heatmap_data = df.pivot_table(index='Eval Script', columns='Optimal Script', values=f'Avg {row_column} Similarity')
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="Blues", linewidths=0.5)
    plt.title(f'EVAL {number_eval} - Heatmap of Avg {row_column} Similarity')
    plt.xlabel('Optimal Script')
    plt.ylabel(f'{eval_optimal} Script')
    plt.savefig(log_file)

################ BLEU EVALUATION

def bleu_algorithm_logical_surface(file_path_references, file_path_candidates, n_gram):
    # References
    with open(file_path_references, 'r') as reference_file:
        references_content = reference_file.read()
    references_pattern = r'<a script\.(\d+) type=DSC>\s*<u speaker=HUM>(.*?)</u>\s*<u speaker=BOT>(.*?)</u>\s*</a>'
    references_matches = re.findall(references_pattern, references_content, re.DOTALL)
    prompt_references_dict = {}
    for reference_match in references_matches:
        hum_text = reference_match[1]
        bot_text = reference_match[2]
        if hum_text not in prompt_references_dict:
            prompt_references_dict[hum_text] = set()
        prompt_references_dict[hum_text].add(bot_text)

    # Candidates
    with open(file_path_candidates, 'r') as candidate_file:
        candidate_content = candidate_file.read()
    candidate_pattern = r">> Prompt: (.*?)\s+>> Response: (.*?)\n\n"
    candidate_matches = re.findall(candidate_pattern, candidate_content, re.DOTALL)
    prompt_candidates_dict = {}
    for candidate_match in candidate_matches:
        prompt, candidate = candidate_match
        if prompt in prompt_candidates_dict:
            prompt_candidates_dict[prompt].append(candidate)
        else:
            prompt_candidates_dict[prompt] = [candidate]

    # BLEU score
    bleu_scores=[]
    weights = tuple([1.0 if i == 0 else 0.0 for i in range(n_gram)])
    for prompt in prompt_candidates_dict.keys():
        if prompt in prompt_references_dict:
            candidate_responses = prompt_candidates_dict[prompt]
            reference_responses = prompt_references_dict[prompt]

            tokenized_references = [response.split() for response in reference_responses]
            tokenized_candidates = [response.split() for response in candidate_responses]

            candidate_bleu_scores = []

            for candidate in tokenized_candidates:
                score = sentence_bleu(tokenized_references, candidate, weights=weights)
                candidate_bleu_scores.append(score)
            highest_score = max(candidate_bleu_scores)
            bleu_scores.append(highest_score)

    if bleu_scores:
        average_bleu_score = sum(bleu_scores) / len(bleu_scores)
    else:
        average_bleu_score = 0
    return average_bleu_score

def clean_list(data):
    cleaned = []
    for sublist in data:
        if any(word.startswith('<u') or '(' in word or ')' in word or 'speaker=' in word for word in sublist):
            continue
        cleaned.append(sublist)
    return cleaned

def bleu_algorithm(script_id, file_path_references, file_path_candidates, n_gram, sandwich_flag=None):
    # References (I am extracting the utterances based on the script_id I am evaluating from) OPTIMAL
    with open(file_path_references, 'r') as reference_file:
        references_content = reference_file.read()

    if not sandwich_flag: # Surface
        scripts = re.findall(r"(<script\.\d+ type=CONV>.*?</script\.\d+>)", references_content, re.DOTALL)
    if sandwich_flag:
        scripts = re.findall(r"(<a script.\d+ type=SDW>.*?</a>)", references_content, re.DOTALL)

    # Getting the utterances only from the selected situation
    if script_id != 0:
        script_content_reference = scripts[script_id-1]
    if script_id == 0:
        script_content_reference = references_content
    references_pattern = r'<u speaker=[^>]*>(.*?)</u>'
    references_matches = re.findall(references_pattern, script_content_reference, re.DOTALL)
    tokenized_references = [reference.split() for reference in references_matches]

    # Candidates RESPONSES BOT
    with open(file_path_candidates, 'r') as candidate_file:
        candidate_content = candidate_file.read()
    candidate_pattern = r">> Response: (.*?)\n\n"
    candidate_matches = re.findall(candidate_pattern, candidate_content, re.DOTALL)
    tokenized_candidates = [candidate.split() for candidate in candidate_matches]
    # if sandwich_flag:
    #     tokenized_candidates = clean_list(tokenized_candidates)

    # # BLEU score
    bleu_scores=[]
    weights = tuple([1.0 if i == 0 else 0.0 for i in range(n_gram)])
    for candidate in tokenized_candidates:

        candidate_bleu_scores = []
        references = []

        if sandwich_flag:
            if any(word.startswith('<u') or '(' in word or ')' in word or 'speaker=' in word for word in candidate):
                score = 0
            else:
                for reference in tokenized_references:
                    score = sentence_bleu([reference], candidate, weights=weights)
                    references.append(reference)
            candidate_bleu_scores.append(score)
        else:
            for reference in tokenized_references:
                score = sentence_bleu([reference], candidate, weights=weights)  # Compare candidate with one reference at a time
                candidate_bleu_scores.append(score)
                references.append(reference)

        highest_value = max(candidate_bleu_scores)
        bleu_scores.append(highest_value)

    if bleu_scores:
        average_bleu_score = sum(bleu_scores) / len(bleu_scores)
    else:
        average_bleu_score = 0
    return average_bleu_score

############# LOGICAL
#############
############# CREATE THE EVALUATION VECTORIAL SPACES FILES
#
# print('Vectorial Spaces Evaluation - Logic Data, creation of vectorial spaces, evaluating them and storing')
#
# if not os.path.exists(os.path.dirname('./data/data_analysis/logical/evaluations/optimal_1.txt')):
#     os.makedirs(os.path.dirname('./data/data_analysis/logical/evaluations/optimal_1.txt'))
#
# open('./data/data_analysis/logical/evaluations/optimal_1.txt', 'w').close()
# for script_id_eval in range (0,13):
#     for script_id_optimal in range(0,13):
#         evaluation_with_vectorial_space(1, script_id_eval, script_id_optimal, log_file='./data/data_analysis/logical/evaluations/optimal_1.txt', optimal_to_optimal=True)
#
# open('./data/data_analysis/logical/evaluations/optimal_2.txt', 'w').close()  # This clears the file by opening & immediately closing it
# for script_id_eval in range (0,13):
#     for script_id_optimal in range(0,13):
#         evaluation_with_vectorial_space(2, script_id_eval, script_id_optimal, log_file='./data/data_analysis/logical/evaluations/optimal_2.txt', optimal_to_optimal=True)
#
# open('./data/data_analysis/logical/evaluations/evaluation_1.txt', 'w').close()
# for script_id_eval in range (0,13):
#     for script_id_optimal in range(0,13):
#         evaluation_with_vectorial_space(1, script_id_eval, script_id_optimal, log_file='./data/data_analysis/logical/evaluations/evaluation_1.txt')
#
# open('./data/data_analysis/logical/evaluations/evaluation_2.txt', 'w').close()  # This clears the file by opening & immediately closing it
# for script_id_eval in range (0,13):
#     for script_id_optimal in range(0,13):
#         evaluation_with_vectorial_space(2, script_id_eval, script_id_optimal, log_file='./data/data_analysis/logical/evaluations/evaluation_2.txt')

########## GRAPHICAL REPRESENTATIONS
##############
############### CREATE THE CSVs FROM THE EVALUATION FILES
#
# print('Vectorial Spaces Evaluation - Logic Data, comparing the spaces and storing')
#
# if not os.path.exists(os.path.dirname('./data/data_analysis/logical/dataframes/optimal_1.csv')):
#     os.makedirs(os.path.dirname('./data/data_analysis/logical/dataframes/optimal_1.csv'))
#
# open('./data/data_analysis/logical/dataframes/optimal_1.csv', 'w').close()
# make_csv_from_data('./data/data_analysis/logical/evaluations/optimal_1.txt', './data/data_analysis/logical/dataframes/optimal_1.csv')
# open('./data/data_analysis/logical/dataframes/optimal_2.csv', 'w').close()
# make_csv_from_data('./data/data_analysis/logical/evaluations/optimal_2.txt', './data/data_analysis/logical/dataframes/optimal_2.csv')
# open('./data/data_analysis/logical/dataframes/evaluation_1.csv', 'w').close()
# make_csv_from_data('./data/data_analysis/logical/evaluations/evaluation_1.txt', './data/data_analysis/logical/dataframes/evaluation_1.csv')
# open('./data/data_analysis/logical/dataframes/evaluation_2.csv', 'w').close()
# make_csv_from_data('./data/data_analysis/logical/evaluations/evaluation_2.txt', './data/data_analysis/logical/dataframes/evaluation_2.csv')

####################### CREATE THE HEATMAPS
#
# print('Vectorial Spaces Evaluation - Logic Data, heatmaps creation')
#
# if not os.path.exists(os.path.dirname('./data/data_analysis/logical/images/row_optimal_1.png')):
#     os.makedirs(os.path.dirname('./data/data_analysis/logical/images/row_optimal_1.png'))
#
# heatmap('./data/data_analysis/logical/dataframes/optimal_1.csv', 'Row', 'Optimal', '1', './data/data_analysis/logical/images/row_optimal_1.png')
# heatmap('./data/data_analysis/logical/dataframes/optimal_1.csv', 'Column', 'Optimal', '1', './data/data_analysis/logical/images/col_optimal_1.png')
#
# heatmap('./data/data_analysis/logical/dataframes/optimal_2.csv', 'Row', 'Optimal', '2', './data/data_analysis/logical/images/row_optimal_2.png')
# heatmap('./data/data_analysis/logical/dataframes/optimal_2.csv', 'Column', 'Optimal', '2', './data/data_analysis/logical/images/col_optimal_2.png')
#
# heatmap('./data/data_analysis/logical/dataframes/evaluation_1.csv', 'Row', 'Eval', '1', './data/data_analysis/logical/images/row_eval_1.png')
# heatmap('./data/data_analysis/logical/dataframes/evaluation_1.csv', 'Column', 'Eval', '1', './data/data_analysis/logical/images/col_eval_1.png')
#
# heatmap('./data/data_analysis/logical/dataframes/evaluation_2.csv', 'Row', 'Eval', '2', './data/data_analysis/logical/images/row_eval_2.png')
# heatmap('./data/data_analysis/logical/dataframes/evaluation_2.csv', 'Column', 'Eval', '2', './data/data_analysis/logical/images/col_eval_2.png')

############# LOGICAL TO SURFACE

# print('BLEU Algorithm Evaluation - Logical to Surface Data \n')
# n_gram_value=1
#
# for script_id_reference in range (0,13):
#    score = bleu_algorithm_logical_surface('./data/training/permuted_logical_to_surface.txt',
#                           f'./data/evaluation_data/evaluation_logic_to_surface/evaluation_situation{script_id_reference}.txt',
#                           n_gram_value)
#    print(f'Average BLEU score, n-grams {n_gram_value} for script id {script_id_reference} is {score}')

########## SURFACE

# print('\n BLEU Algorithm Evaluation - Surface Data \n')
# n_gram_value=1
#
# for script_id_reference in range (0,13):
#     score = bleu_algorithm(script_id_reference, './data/training/prompt_surface.txt',
#                                            f'./data/evaluation_data/evaluation_surface/evaluation_situation{script_id_reference}.txt',
#                                            n_gram_value)
#     print(f'Average BLEU score, n-grams {n_gram_value} for script id {script_id_reference} is {score}')

############### SANDWICH

# print('\n BLEU Algorithm Evaluation - Sandwich Data \n')
# n_gram_value=1
#
# for script_id_reference in range (0,13):
#     score = bleu_algorithm(script_id_reference, './data/training/prompt_sandwich.txt',
#                                            f'./data/evaluation_data/emma_sandwich/evaluation_situation{script_id_reference}.txt',
#                                            n_gram_value, sandwich_flag=True)
#     print(f'Average BLEU score, n-grams {n_gram_value} for script id {script_id_reference} is {score}')