import pandas as pd
import os
import re 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

path = "wiki-qa-data/"

def analyse_data(path, rouges):
  for file in os.listdir(path):
      if file.endswith(".tsv") and "-ans" in file:
          origin = file.split('.')[0].split("-")[1]
          data = pd.read_csv(path + file, sep="\t")

          # Ensure the dataset has the required columns
          if 'verbosity_control' not in data.columns or 'rouge_scores' not in data.columns:
              raise ValueError("Dataset must contain 'verbosity_control' and 'rouge_scores' columns.")

          # Group by verbosity_control and aggregate rouge_scores, similarity, cosine_similarity columns
          grouped_data = data.groupby('verbosity_control')['rouge_scores'].apply(list).reset_index()
          grouped_data['similarity'] = data.groupby('verbosity_control')['similarity'].apply(list).reset_index()['similarity']
          grouped_data['length_ratio'] = data.groupby('verbosity_control')['length_ratio'].apply(list).reset_index()['length_ratio']
          grouped_data['bert_scores'] = data.groupby('verbosity_control')['bert_scores'].apply(list).reset_index()['bert_scores']

          # Extract F1 scores (fmeasure) from the ROUGE SCORE strings
          def extract_f1_scores(rouge_scores_list, rougeBool=False, bertBool=False):
              f1_scores = {rougeN: [] for rougeN in rouges}
              for rouge_scores in rouge_scores_list:
                  try:
                      rouge_scores_dict = eval(rouge_scores)  # Convert string to dictionary
                      if rougeBool:
                          for rougeN in rouges:
                              if rougeN in rouge_scores_dict:
                                  f1_scores[rougeN].append(rouge_scores_dict[rougeN]['f1'])
                      elif bertBool:
                          f1_scores['bert'] = [score['f1'] for score in rouge_scores_dict.values()]
                  except Exception as e:
                      print(f"Error processing rouge_scores: {rouge_scores}. Error: {e}")
                      for rougeN in rouges:
                          f1_scores[rougeN].append(0)  # Append 0 in case of error
              return f1_scores

          # EXTRACT ROUGE F1 SCORES
          rogue_f1_scores = grouped_data['rouge_scores'].apply(lambda x: extract_f1_scores(x, rougeBool=True))
          for rougeN in rouges:
              grouped_data[f'{rougeN}_f1_scores'] = rogue_f1_scores.apply(lambda x: x[rougeN])
              grouped_data[f'{rougeN}_f1_avg'] = grouped_data[f'{rougeN}_f1_scores'].apply(lambda x: sum(x) / len(x) if x else 0)
              grouped_data[f'{rougeN}_f1_std'] = grouped_data[f'{rougeN}_f1_scores'].apply(
                    lambda x: (sum((score - (sum(x) / len(x)))**2 for score in x) / len(x))**0.5 if x else 0
                )
              grouped_data[f'{rougeN}_f1_max'] = grouped_data[f'{rougeN}_f1_scores'].apply(lambda x: max(x) if x else 0)
              grouped_data[f'{rougeN}_f1_max_index'] = grouped_data[f'{rougeN}_f1_scores'].apply(lambda x: x.index(max(x)) if x else -1)
              grouped_data[f'{rougeN}_f1_max_top3'] = grouped_data[f'{rougeN}_f1_scores'].apply(lambda x: (sorted(x, reverse=True)[:3]) if x else 0)
              grouped_data[f'{rougeN}_f1_max_top3_index'] = grouped_data[f'{rougeN}_f1_max_top3'].apply(lambda x: [x.index(score) for score in x] if x else -1)
              grouped_data[f'{rougeN}_f1_max_top3_answer'] = grouped_data[f'{rougeN}_f1_max_top3_index'].apply(lambda x: [data.iloc[i]['answer'] for i in x] if x != -1 else None)

          # EXTRACT BERT SCORES F1 SCORES
          bert_f1_scores = grouped_data['rouge_scores'].apply(lambda x: extract_f1_scores(x, bertBool=True))
          bert_f1_scores = bert_f1_scores.apply(lambda x: x['bert'])
          grouped_data['bert_f1_scores'] = bert_f1_scores
          grouped_data['bert_f1_avg'] = bert_f1_scores.apply(lambda x: sum(x) / len(x) if x else 0)
          grouped_data['bert_f1_std'] = bert_f1_scores.apply(
              lambda x: (sum((score - (sum(x) / len(x)))**2 for score in x) / len(x))**0.5 if x else 0
          )
          grouped_data['bert_f1_max'] = bert_f1_scores.apply(lambda x: max(x) if x else 0)
          grouped_data['bert_f1_max_index'] = bert_f1_scores.apply(lambda x: x.index(max(x)) if x else -1)
          grouped_data['bert_f1_max_top3'] = bert_f1_scores.apply(lambda x: (sorted(x, reverse=True)[:3]) if x else 0)
          grouped_data['bert_f1_max_top3_index'] = grouped_data['bert_f1_max_top3'].apply(lambda x: [x.index(score) for score in sorted(x, reverse=True)[:3]] if x else -1)
          grouped_data['bert_f1_max_top3_answer'] = grouped_data['bert_f1_max_top3_index'].apply(lambda x: [data.iloc[i]['answer'] for i in x] if x != -1 else None)

          # EXTRACT TOP 3 similarity scores
          grouped_data['similarity_top3'] = grouped_data['similarity'].apply(lambda x: (sorted(x, reverse=True)[:3]) if x else 0)
          grouped_data['similarity_top3_index'] = grouped_data['similarity'].apply(lambda x: [x.index(score) for score in sorted(x, reverse=True)[:3]] if x else -1)
          grouped_data['similarity_top3_answer'] = grouped_data['similarity_top3_index'].apply(lambda x: [data.iloc[i]['answer'] for i in x] if x != -1 else None)

          print(f"\nData Analysis for: {origin}")
          grouped_data['origin'] = origin
          grouped_data.pop('rouge_scores')
          print(grouped_data)

          # Save to CSV
          grouped_data.to_csv(path + f"analysis.csv", mode='a', header=not os.path.exists(path + f"analysis.csv"), index=False)
          print(f"Analysis appended to: {path}analysis.csv")

def display_metric_scores(path, metrics):
    """
    Display KDE plots for multiple metrics (e.g., ROUGE, BERT, similarity) grouped by non-unique verbosity control keys [merge all rows with same key].
    
    Parameters:
    - path: Path to the CSV file containing the analysis data.
    - metrics: A dictionary where keys are metric groups (e.g., "rouge", "similarity") and values are:
               - A dictionary with:
                   - "metrics": A list of dictionaries for grouped subplots (e.g., [{"rouge2": "column"}, {"rougeL": "column"}])
                   - "xlim" (optional): A tuple specifying the x-axis limits (e.g., (-0.3, 1)).
    """
    df = pd.read_csv(path)
    verbosity_controls = df['verbosity_control'].unique()

    for metric_group, group_info in metrics.items():
        group_metrics = group_info["metrics"]
        xlim = group_info.get("xlim", None)  # Get xlim if provided, otherwise default to None

        if isinstance(group_metrics, list):  # Multiple subplots for this metric group
            num_subplots = len(group_metrics)
            subplots_per_row = 2
            num_rows = (num_subplots + subplots_per_row - 1) // subplots_per_row  # Calculate the number of rows

            # Create a figure for this metric group
            fig, axes = plt.subplots(num_rows, subplots_per_row, figsize=(12, 6 * num_rows), sharey=True)
            axes = axes.flatten()  # Flatten the axes array for easier indexing

            # Hide unused subplots if the number of metrics is less than the total subplots
            for ax in axes[num_subplots:]:
                ax.axis("off")

            for ax, metric_dict in zip(axes, group_metrics):
                for metric, column in metric_dict.items():
                    _plot_metric(df, verbosity_controls, metric, column, ax, xlim)
                    ax.set_title(f"{metric.upper()} Scores")

            # Add a legend for this figure
            handles, labels = axes[0].get_legend_handles_labels()  # Collect legend from the first subplot
            fig.legend(handles, labels, loc="upper center", ncol=2, title="Verbosity Control")
            plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust layout to make space for the legend
            plt.subplots_adjust(top=0.85 if subplots_per_row == 2 else 0.7) 
            plt.show()

        else:  # Single plot for this metric group
            fig, ax = plt.subplots(figsize=(8, 6))
            _plot_metric(df, verbosity_controls, metric_group, group_metrics, ax, xlim)
            ax.set_title(f"{metric_group.upper()} Scores")

            # Add a legend for this figure
            handles, labels = ax.get_legend_handles_labels()  # Collect legend from the single plot
            fig.legend(handles, labels, loc="upper center", ncol=2, title="Verbosity Control")
            plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust layout to make space for the legend
            plt.subplots_adjust(top=0.7)  # Further adjust the top margin for the legend
            plt.show()


def _plot_metric(df, verbosity_controls, metric, column, ax, xlim):
    """
    Helper function to plot a single metric on a given axis with optional x-axis limits.
    """
    averages = {}
    colors = {}

    for control in verbosity_controls:
        subset = df[df['verbosity_control'] == control][column].dropna()

        # Convert string representations of lists to actual lists (if applicable)
        if column.endswith("_f1_scores") or column in ["similarity", "length_ratio"]:
            subset = subset.apply(eval).explode().astype(float)

        if len(subset) > 1:  # Only plot if we have enough data
            kde = sns.kdeplot(subset, label=control, fill=False, ax=ax)
            colors[control] = kde.get_lines()[-1].get_color()  # Store the color used for this control
            averages[control] = subset.mean()

    # Draw vertical lines for the averages with the same colors as the KDE plots
    for control, avg in averages.items():
        ax.axvline(avg, linestyle="--", color=colors[control], label=f"{control} Avg: {avg:.2f}")

    ax.set_xlabel(f"{metric.capitalize()} Score")
    if xlim:  # Apply x-axis limits if provided
        ax.set_xlim(xlim)
    ax.grid()

if __name__ == "__main__":
    rouges = ["rouge2", "rougeL"]
    
    analyse_data(path, rouges)
    print("Analysis completed.")

    metrics = {
        "f1": {
            "metrics": [
                # {"rouge2 f1": "rouge2_f1_scores"},
                {"rougeL f1": "rougeL_f1_scores"},
                {"bert f1": "bert_f1_scores"},
                {"similarity": "similarity"}
            ],
            "xlim": (-0.3, 1)
        },
        # "length_ratio": {
        #     "metrics": [{"length_ratio": "length_ratio"}],
        #     "xlim": (0, 5)
        # }
    }

    display_metric_scores(path + "analysis.csv", metrics)