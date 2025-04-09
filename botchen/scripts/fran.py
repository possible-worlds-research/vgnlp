# The script takes the data extracted from VG and applies words permutations from their synonyms or hypernyms. 
# These variations increase the generalization capability of Botchen.
# It also create a prompt file for automatic evaluation.

# **Usage:** ```python3 ./scripts/permutations_prompts.py```.
# Takes  **`./data/extracted_scripts.txt`** as input. Gives **`./data/training_data/*'** and **`./data/prompt_file.txt'** as output.
# The user can change the words to substitute at the end of the file.

import re
import os
import argparse
import random
import nltk
from nltk.corpus import wordnet as wn
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu

# Download required resources
# nltk.download('wordnet')
# nltk.download('omw-1.4')  # For wordnet synonyms in different languages

################# Logical/Surface Formats

def bleu_algorithm(file_path_references, file_path_candidates):
    # References
    with open(f'./data/training/{file_path_references}', 'r') as reference_file:
        references_content = reference_file.read()
    references_pattern = r'<a script\.(\d+) type=DSC>\s*<u speaker=HUM>(.*?)</u>\s*<u speaker=BOT>(.*?)</u>\s*</a>'
    references_matches = re.findall(references_pattern, references_content, re.DOTALL)
    prompt_references_dict = {}
    for reference_match in references_matches:
        script_num = int(reference_match[0])  # Get the script number
        hum_text = reference_match[1]  # Get the human text
        bot_text = reference_match[2]
        if hum_text not in prompt_references_dict:
            prompt_references_dict[hum_text] = set()
        prompt_references_dict[hum_text].add(bot_text)

    # Candidates
    with open(f'./data/{file_path_candidates}', 'r') as candidate_file:
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
    for prompt in prompt_candidates_dict.keys():
        if prompt in prompt_references_dict:
            # Get the candidate responses and reference responses for the matching prompt
            candidate_responses = prompt_candidates_dict[prompt]
            reference_responses = prompt_references_dict[prompt]

            # Tokenize responses (split by words)
            tokenized_references = [response.split() for response in reference_responses]
            tokenized_candidates = [response.split() for response in candidate_responses]

            # # For each candidate response, compute the BLEU score
            for candidate in tokenized_candidates:
                score = sentence_bleu(tokenized_references, candidate)
                bleu_scores.append(score)
                print(candidate, tokenized_references, score)

    if bleu_scores:
        average_bleu_score = sum(bleu_scores) / len(bleu_scores)
    else:
        average_bleu_score = 0
    print(f'Average BLEU score is {average_bleu_score}')
    return average_bleu_score

bleu_algorithm('extracted_logical_to_surface.txt', 'emma_evaluation_logic_to_surface/evaluation_situation1.txt')