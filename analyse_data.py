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

        # Extract F1 scores (fmeasure) from the ROUGE SCORE similarity strings
        rouges = ["rouge2", "rougeL"]

        def extract_f1_scores(similarity_list):
          rouges_f1_scores = []
          for rougeN in rouges:
            rougeN_f1_scores = []
            for similarity in similarity_list:
                # Use regex to extract the fmeasure value for rouge1
                match = re.search(fr"'{rougeN}':.*?fmeasure=(\d+\.\d+)", similarity)
                if match:
                    rougeN_f1_scores.append(float(match.group(1)))  # Extract and convert to float
            rouges_f1_scores.append(rougeN_f1_scores)
          return rouges_f1_scores

        rouges_f1_scores = grouped_data['similarity'].apply(extract_f1_scores)
        for rougeN, f1_scores in zip(rouges, zip(*rouges_f1_scores)):
          grouped_data[f'{rougeN}_f1_scores'] = f1_scores
          grouped_data[f'{rougeN}_f1_avg'] = grouped_data[f'{rougeN}_f1_scores'].apply(lambda x: sum(x) / len(x) if x else 0)
          grouped_data[f'{rougeN}_f1_max'] = grouped_data[f'{rougeN}_f1_scores'].apply(lambda x: max(x) if x else 0)
          grouped_data[f'{rougeN}_f1_max_index'] = grouped_data[f'{rougeN}_f1_scores'].apply(lambda x: x.index(max(x)) if x else -1)

        print(f"\nData Analysis for: {origin}")
        print(grouped_data)