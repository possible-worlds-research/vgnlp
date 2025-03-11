import pandas as pd
import re

def make_csv_from_data(file_path, name):
    # Sample data (replace this with reading from a file)
    with open(file_path, 'r') as file:
        data = file.read()

    # Regular expressions to extract key information
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
    summary_df.to_csv(f"./data/data_analysis/{name}", index=False)

make_csv_from_data('./data/evaluation_file.txt', 'new_file.csv')
