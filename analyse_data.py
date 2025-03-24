import pandas as pd
import os
import re  # Use regex to extract values from the similarity strings

path = "wiki-qa-data/"

for file in os.listdir(path):
    if file.endswith(".tsv") and "-ans" in file:
        origin = file.split('.')[0].split("-")[1]
        data = pd.read_csv(path + file, sep="\t")

        # Ensure the dataset has the required columns
        if 'verbosity_control' not in data.columns or 'similarity' not in data.columns:
            raise ValueError("Dataset must contain 'verbosity_control' and 'similarity' columns.")

        # Group by verbosity_control and aggregate similarity values
        grouped_data = data.groupby('verbosity_control')['similarity'].apply(list).reset_index()

        # Extract F1 scores (fmeasure) from the similarity strings
        def extract_f1_scores(similarity_list):
            rouge1_f1_scores = []
            rougeL_f1_scores = []
            for similarity in similarity_list:
                # Use regex to extract the fmeasure value for rouge1
                match = re.search(r"'rouge1':.*?fmeasure=(\d+\.\d+)", similarity)
                if match:
                    rouge1_f1_scores.append(float(match.group(1)))  # Extract and convert to float
                match = re.search(r"'rougeL':.*?fmeasure=(\d+\.\d+)", similarity)
                if match:
                    rougeL_f1_scores.append(float(match.group(1)))  # Extract and convert to float
            return rouge1_f1_scores, rougeL_f1_scores

        grouped_data['rouge1_f1_scores'], grouped_data['rougeL_f1_scores'] = grouped_data['similarity'].apply(extract_f1_scores)

        # Calculate the mean F1 score for each verbosity_control
        grouped_data['mean_rouge1_f1_score'] = grouped_data['rouge1_f1_scores'].apply(lambda x: sum(x) / len(x) if x else 0)
        grouped_data['mean_rougeL_f1_score'] = grouped_data['rougeL_f1_scores'].apply(lambda x: sum(x) / len(x) if x else 0)

        print(grouped_data)