import pandas as pd
import os
import re 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

path = "wiki-qa-data/"
rouges = ["rouge2", "rougeL"]

def analyse_data(path):
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
            grouped_data[f'{rougeN}_f1_std'] = grouped_data[f'{rougeN}_f1_scores'].apply(
                  lambda x: (sum((score - (sum(x) / len(x)))**2 for score in x) / len(x))**0.5 if x else 0
              )
            grouped_data[f'{rougeN}_f1_max'] = grouped_data[f'{rougeN}_f1_scores'].apply(lambda x: max(x) if x else 0)
            grouped_data[f'{rougeN}_f1_max_index'] = grouped_data[f'{rougeN}_f1_scores'].apply(lambda x: x.index(max(x)) if x else -1)
            grouped_data[f'{rougeN}_f1_max_top3'] = grouped_data[f'{rougeN}_f1_scores'].apply(lambda x: (sorted(x, reverse=True)[:3]) if x else 0)
            grouped_data[f'{rougeN}_f1_max_top3_index'] = grouped_data[f'{rougeN}_f1_scores'].apply(lambda x: [x.index(score) for score in sorted(x, reverse=True)[:3]] if x else -1)
            grouped_data[f'{rougeN}_f1_max_top3_answer'] = grouped_data[f'{rougeN}_f1_max_top3_index'].apply(lambda x: [data.iloc[i]['answer'] for i in x] if x != -1 else None)

          print(f"\nData Analysis for: {origin}")
          grouped_data['origin'] = origin
          print(grouped_data)

          # Save to CSV
          grouped_data.to_csv(path + f"analysis.csv", mode='a', header=not os.path.exists(path + f"analysis.csv"), index=False)
          print(f"Analysis appended to: {path}analysis.csv")

def display_graphs(path):
    df = pd.read_csv(path)

    verbosity_controls = df['verbosity_control']

    rouge_columns = {rouge: df.columns[df.columns.str.contains(fr"{rouge}_f1_", regex=True)] for rouge in rouges}
    print("ROUGE Columns:", rouge_columns)

    # Create a figure with subplots for side-by-side plots
    fig, axes = plt.subplots(1, len(rouges), figsize=(16, 6), sharey=True)

    for i, (rouge, rouge_col) in enumerate(rouge_columns.items()):
        # Only take columns that contain f1_scores
        score_column = f"{rouge}_f1_scores"
        print(f"Processing ROUGE metric: {rouge}")

        ax = axes[i]  # Select the subplot for this ROUGE metric

        # Loop through each verbosity control and plot its distribution
        averages = {}
        colors = {}  # Store the colors used for each verbosity control
        for control in verbosity_controls.unique():
            subset = df[df['verbosity_control'] == control][score_column].dropna()

            # Convert string representations of lists to actual lists
            subset = subset.apply(eval)

            # Flatten the list of scores
            subset = subset.explode().astype(float)

            if len(subset) > 1:  # Only plot if we have enough data
                kde = sns.kdeplot(subset, label=control, fill=False, ax=ax)
                # Get the color used by Seaborn for this plot
                print(len(subset), len(kde.get_lines()))
                colors[control] = kde.get_lines()[-1].get_color()

            # Calculate the average F1 score for this verbosity control
            if len(subset) > 0:
                averages[control] = subset.mean()

        # Draw vertical lines for the averages with the same colors as the KDE plots
        for control, avg in averages.items():
            ax.axvline(avg, linestyle="--", color=colors[control], alpha=0.7)

        ax.set_title(f"Distribution of {rouge.upper()} F1 Scores")
        ax.set_xlabel(f"{rouge.upper()} F1 Score")
        ax.set_xlim(-0.3, 1)
        ax.grid()
        if i == 0:
            ax.set_ylabel("Density")
        ax.legend(title="Verbosity Control")

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
     

if __name__ == "__main__":
  # Comment out whichever function you want to run
  # analyse_data(path)
  display_graphs(path+"analysis.csv")