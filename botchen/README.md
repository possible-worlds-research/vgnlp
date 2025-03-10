# VGNLP Botchen

VGNLP Botchen is designed to extract data from the [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) dataset, convert it into a conversational format, and use it to *Botchen* tiny chatbot . It also includes evaluation scripts for assessing the performance of the trained chatbot.

## Scripts Overview

The project includes the following scripts:

- **`./scripts/extract_from_vg.py`**  
  Converts the Visual Genome (VG) data format into a conversational format readable by Botchen. This script extracts situations based on a selected situation ID and stores the data in **`./data/extracted_scripts.txt`**, now suitable for Botchen training.

- **`./scripts/permutations_prompts.py`**  
  Applies permutations to the extracted data (from **`./data/extracted_scripts.txt`**) to create varied versions of each situation. The permutations help the chatbot generalize better across different scenarios. This script generates **`./data/training_data/*`** (12 files, format *'ideallanguage_{situational_id}.txt'*) and a prompt file **`./data/prompt_file.txt`**.

- **`./scripts/controllers.py`**  
  Handles automatic conversation generation in Botchen's console for evaluation. This script generates conversations from the prompts in **`./data/prompt_file.txt`** and stores the conversations in **`./data/evaluation_data/evaluation_situation{situational_id}.txt`**.

- **`./scripts/evaluation.py`**  
  Compares Botchen?s responses (evaluation data) with optimal training data (from training) using cosine similarity. It creates vectorial spaces for both the evaluation and training data and evaluates their similarity by row and column.

The user can create the necessary folder structure starting with only the scripts and the **`./data/ideallanguage.zip`** file. The scripts will generate all other required files for evaluation. For simplicity, examples of the desired files and the training data used for evaluation have been provided

***

## Extracting data

The **`./scripts/extract_from_vg.py`** script extracts situations from the Visual Genome dataset, turning them into a conversational format. For instance, a situation in the Visual Genome might look like:

```<situation id=1> <entity id=1058549> tree.n.01(1058549) sparse(1058549) by(1058549,1058534)```

This is converted into a conversational format suitable for training Botchen, like this:

```<script.1 type=CONV> <u speaker=HUM>(tree.n sparse by-sidewalk)</u>```

**Usage:** ```python3 ./scripts/extract_from_vg.py```. Takes  **`./data/ideallanguage.txt`** as input. Gives **`./data/extracted_scripts.txt'** as output. The user can change the VG situation ID which to revert to Botchen format at the end of the file. Now we selected 12 scripts, covering diverse topics.

## Make up the data for the training 

The **`./scripts/permutations_prompts.py`** script takes the extracted data stored in **`./data/extracted_scripts.txt`** and applies permutations to create different versions of each situation. For example, it may replace specific nouns (e.g., replacing 'zebra' with more general terms like 'animal') or swap synonyms. These variations increase the generalization capability of Botchen.

The output consists of 12 files in **`./data/training_data/*`**, with 9 different versions of each situational script. These are used as training input for Botchen.

## Evaluation prompts and conversation

The **`./scripts/permutations_prompts.py`** script creates a prompt file **`./data/prompt_file.txt`**. This file contains all the situations and their permutations, providing the inputs to evaluate Botchen's performance (each utterance of the chosen situation repeated 4 times)

**Usage:** ```python3 ./scripts/permutations_prompts.py```. Takes  **`./data/extracted_scripts.txt`** as input. Gives **`./data/training_data/*'** and **`./data/prompt_file.txt'** as output. The user can change the words to substitute at the end of the file.

The **`./scripts/controllers.py`** script enables the generation of a conversation using Botchen. It uses the *make_evaluating_conversation function*, which takes the prompt file and outputs a conversation. This conversation is stored in **`./data/evaluation_data/evaluation_situation{script_id}.txt`** for evaluation.

**Usage:** (on Bash Botchen console) ```flask training evaluation_with_vectorial_space chat en 2 ./data/chat/en/prompt_file.txt 1 print_statement log_file ```. Takes  **`./data/prompt_file.txt'** as input. Gives **`./data/evaluation_data/*'** as output. The user can dinamically select the situation in which to test the model on. 

## Evaluation

The .**`/scripts/evaluation.py`** script evaluates the conversation output by comparing it with the optimal training data. The script creates vectorial spaces for both the evaluation and training data, with entities as rows and properties of them as columns. It then compares the two spaces using cosine similarity. Two frameworks are used for comparison. The first fits the evaluating space to the optimal space by adding missing dimensions (filled with zeros). The second matches the intersecting dimensions between the two spaces and compares them. Cosine similarity is computed for both rows and columns.

**Usage:** (A) Evaluating framework: ```python3 ./scripts/evaluation.py evaluation 1 2 2```. Evaluates within first evaluation framework, evaluating situation 2, optimal situation 2. (B) Create matrices:   ```python3 ./scripts/evaluation.py create_matrices 1 --optimal_script --saving_directory './data/vectorial_spaces/optimal/'```. Evaluating  with situation 1, from training (otpimal) data and saving the space in the directory. If one wants to make them from evaluating data: ```python3 ./scripts/evaluation.py create_matrices 2 --saving_directory './data/vectorial_spaces/evaluation/'```. In general, the user will need **`./data/evaluation_data/*'** as input and generates './data/vectorial_spaces/evaluation/*' or './data/vectorial_spaces/optimal/*' as csv outputs. The user can dinamically select the situation in which to test the model on and if to store it.

## Requirements

Python 3.11
