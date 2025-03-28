import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def analyse_data(path, analysis_file, rouges):
  for file in os.listdir(path):
      if file.endswith(".tsv") and "-ans" in file:
          origin = file.split('.')[0].split("-")[1]
          data = pd.read_csv(path + file, sep="\t")

          # Ensure the dataset has the required columns
          if 'verbosity_control' not in data.columns or 'rouge_scores' not in data.columns:
              raise ValueError("Dataset must contain 'verbosity_control' and 'rouge_scores' columns.")

          # Group by verbosity_control and aggregate rouge_scores, similarity, cosine_similarity columns
          grouped_data = data.groupby('verbosity_control')['rouge_scores'].apply(list).reset_index()
          grouped_data['similarity_scores'] = data.groupby('verbosity_control')['similarity'].apply(list).reset_index()['similarity']
          grouped_data['length_ratio_scores'] = data.groupby('verbosity_control')['length_ratio'].apply(list).reset_index()['length_ratio']
          grouped_data['bert_scores'] = data.groupby('verbosity_control')['bert_scores'].apply(list).reset_index()['bert_scores']
          grouped_data['answer'] = data.groupby('verbosity_control')['answer'].apply(list).reset_index()['answer']

          # Extract F1 scores (fmeasure) from the ROUGE SCORE strings
          def extract_f1_scores(scores_list, rougeBool=False, bertBool=False):
              f1_scores = {"bert": []}
              f1_scores.update({rougeN: [] for rougeN in rouges})
              for scores in scores_list:
                  try:
                      scores_dict = eval(scores)  # Convert string to dictionary
                      if rougeBool:
                          for rougeN in rouges:
                              if rougeN in scores_dict:
                                  f1_scores[rougeN].append(scores_dict[rougeN]['f1'])
                      elif bertBool:
                          f1_scores['bert'].append(scores_dict['f1'])
                  except Exception as e:
                      print(f"Error processing rouge_scores: {scores}. Error: {e}")
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
            grouped_data[f'{rougeN}_f1_top3'] = grouped_data[f'{rougeN}_f1_scores'].apply(lambda x: (sorted(x, reverse=True)[:3]) if x else 0)
            grouped_data[f'{rougeN}_f1_top3_index'] = grouped_data[f'{rougeN}_f1_scores'].apply(lambda x: [x.index(score) for score in sorted(x, reverse=True)[:3]] if x else -1)
            grouped_data[f'{rougeN}_f1_top3_answer'] = grouped_data.apply(
                lambda row: [row['answer'][i] for i in row[f'{rougeN}_f1_top3_index']] if row[f'{rougeN}_f1_top3_index'] != -1 else None,
                axis=1
            )
          
          # EXTRACT BERT SCORES F1 SCORES
          bert_f1_scores = grouped_data['bert_scores'].apply(lambda x: extract_f1_scores(x, bertBool=True))
          print(bert_f1_scores[0])
          bert_f1_scores = bert_f1_scores.apply(lambda x: x['bert'])
          grouped_data['bert_f1_scores'] = bert_f1_scores
          grouped_data['bert_f1_avg'] = bert_f1_scores.apply(lambda x: sum(x) / len(x) if x else 0)
          grouped_data['bert_f1_std'] = bert_f1_scores.apply(
              lambda x: (sum((score - (sum(x) / len(x)))**2 for score in x) / len(x))**0.5 if x else 0
          )
          grouped_data['bert_f1_max'] = bert_f1_scores.apply(lambda x: max(x) if x else 0)
          grouped_data['bert_f1_max_index'] = bert_f1_scores.apply(lambda x: x.index(max(x)) if x else -1)
          grouped_data['bert_f1_top3'] = bert_f1_scores.apply(lambda x: (sorted(x, reverse=True)[:3]) if x else 0)
          grouped_data['bert_f1_top3_index'] = grouped_data['bert_f1_scores'].apply(lambda x: [x.index(score) for score in sorted(x, reverse=True)[:3]] if x else -1)
          grouped_data['bert_f1_top3_answer'] = grouped_data.apply(
                lambda row: [row['answer'][i] for i in row['bert_f1_top3_index']] if row['bert_f1_top3_index'] != -1 else None,
                axis=1
          )
          # EXTRACT TOP 3 similarity scores
          grouped_data['similarity_top3'] = grouped_data['similarity_scores'].apply(lambda x: (sorted(x, reverse=True)[:3]) if x else 0)
          grouped_data['similarity_top3_index'] = grouped_data['similarity_scores'].apply(lambda x: [x.index(score) for score in sorted(x, reverse=True)[:3]] if x else -1)
          grouped_data['similarity_top3_answer'] = grouped_data.apply(
                lambda row: [row['answer'][i] for i in row['similarity_top3_index']] if row['similarity_top3_index'] != -1 else None,
                axis=1
            )

          print(f"\nData Analysis for: {origin}")
          grouped_data['origin'] = origin
          grouped_data.pop('rouge_scores')
          print(grouped_data)

          # Save to CSV
          grouped_data.to_csv(f"{path}{analysis_file}", mode='a', header=not os.path.exists(path + f"analysis.csv"), index=False)
          print(f"Analysis appended to: {path}{analysis_file}")

def save_top_index_counts_to_csv(df, verbosity_controls, metrics, output_file):
    """
    Saves the top 3 indices and their counts for each metric into a CSV file with separate columns for each top index and count.
    """
    csv_data = []

    for metric in metrics:
        for control in verbosity_controls:
            column = f"{metric}_top3_index"
            if column in df.columns:
                indices = df[df['verbosity_control'] == control][column].dropna()
                answers = df[df['verbosity_control'] == control][f"{metric}_top3_answer"].dropna()
                
                all_indices = indices.apply(eval).reset_index(drop=True)
                all_answers = answers.apply(eval).reset_index(drop=True)

                subset = pd.concat([all_indices, all_answers], axis=1)

                all_indices_answers = subset.apply(lambda x: list(zip(x[0], x[1])), axis=1)
                all_indices_answers = all_indices_answers.apply(lambda x: sorted(x, key=lambda y: y[0], reverse=True))
                all_indices = all_indices_answers.explode().apply(lambda x: x[0])
                index_counts = all_indices.value_counts().sort_values(ascending=False)

                top_indices = index_counts.index.tolist()[:3]
                # Fetch all the answers corresponding to the top indices from all files
                top_answers = []
                for index in top_indices:
                    top_answers.append(all_indices_answers.apply(lambda x: [answer[1] for answer in x if answer[0] == index]).tolist())
                top_counts = index_counts.tolist()[:3]

                # Pad with "N/A" if there are fewer than 3 indices
                while len(top_indices) < 3:
                    top_indices.append("N/A")
                    top_counts.append("N/A")
                    top_answers.append("N/A")

                # Add a row to the CSV data
                csv_data.append({
                    "Verbosity Control": control,
                    "Metric": metric,
                    "Top1 Index": top_indices[0],
                    "Top1 Count (all files)": top_counts[0],
                    "Top1 Answers from each file": top_answers[0],
                    "Top2 Index": top_indices[1],
                    "Top2 Count (all files)": top_counts[1],
                    "Top2 Answers from each file": top_answers[1],
                    "Top3 Index": top_indices[2],
                    "Top3 Count (all files)": top_counts[2],
                    "Top3 Answers from each file": top_answers[2],
                })
            else:
                # If the column doesn't exist, add a row with "N/A"
                csv_data.append({
                    "Verbosity Control": control,
                    "Metric": metric,
                    "Top1 Index": "N/A",
                    "Top1 Count (all files)": "N/A",
                    "Top1 Answers from each file": "N/A",
                    "Top2 Index": "N/A",
                    "Top2 Count (all files)": "N/A",
                    "Top2 Answers from each file": "N/A",
                    "Top3 Index": "N/A",
                    "Top3 Count (all files)": "N/A",
                    "Top3 Answers from each file": "N/A",
                })

    # Convert the data to a DataFrame
    csv_df = pd.DataFrame(csv_data)

    # Save the DataFrame to a CSV file
    csv_df.to_csv(output_file, index=False)
    print(f"Top 3 index counts saved to {output_file}")

# =========== PLOTTING FUNCTIONS ===========

def display_metric_scores(df, verbosity_controls, metrics, type="scores"):
    """
    Display KDE plots for multiple metrics (e.g., ROUGE, BERT, similarity) grouped by non-unique verbosity control keys [merge all rows with same key].
    
    Parameters:
    - path: Path to the CSV file containing the analysis data.
    - metrics: A dictionary where keys are metric groups (e.g., "rouge", "similarity") and values are:
               - A dictionary with:
                   - "metrics": A list of dictionaries for grouped subplots (e.g., [{"rouge2": "column"}, {"rougeL": "column"}])
                   - "xlim" (optional): A tuple specifying the x-axis limits (e.g., (-0.3, 1)).
    """
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
                    if type == "scores":
                        _plot_metric(df, verbosity_controls, metric, column, ax, xlim)
                    elif type == "top3":
                        _plot_top3_scores(df, verbosity_controls, metric, column, ax, xlim)
                    ax.set_title(f"{metric.upper()} Scores")

            # Add a legend for this figure
            handles, labels = axes[0].get_legend_handles_labels()  # Collect legend from the first subplot
            fig.legend(handles, labels, loc="upper center", ncol=1, title="Verbosity Control")
            plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust layout to make space for the legend
            plt.subplots_adjust(top=0.8 if subplots_per_row == 2 else 0.7) 
            plt.show()

        else:  # Single plot for this metric group
            fig, ax = plt.subplots(figsize=(8, 6))
            _plot_metric(df, verbosity_controls, metric_group, group_metrics, ax, xlim)
            ax.set_title(f"{metric_group.upper()} Scores")

            # Add a legend for this figure
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=1, title="Verbosity Control")
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
        subset = df[df['verbosity_control'] == control][f"{column}_scores"].dropna()

        subset = subset.apply(eval).explode().astype(float) # Convert string representations of lists to actual lists

        if len(subset) > 1:  # Only plot if we have enough data
            kde = sns.kdeplot(subset, label=control, fill=False, ax=ax)
            colors[control] = kde.get_lines()[-1].get_color()  # Store the color used for this control
            averages[control] = subset.mean()

    # Draw vertical lines for the averages with the same colors as the KDE plots
    for control, avg in averages.items():
        ax.axvline(avg, linestyle="--", color=colors[control])

    ax.set_xlabel(f"{metric.capitalize()} Score")
    if xlim:  # Apply x-axis limits if provided
        ax.set_xlim(xlim)
    ax.grid()

def _plot_top3_scores(df, verbosity_controls, metric, column, ax, xlim=None):
    """
    Helper function to plot the top 3 scores for a given metric.
    """
    index_counts = {}
    colors = sns.color_palette("tab10", n_colors=len(verbosity_controls))  # Assign unique colors for each verbosity control

    for i, control in enumerate(verbosity_controls):
        subset = df[df['verbosity_control'] == control][f"{column}_top3_index"].dropna()

        # Flatten the list of indices and count occurrences
        all_indices = subset.apply(eval).explode().astype(int)  # Convert string representations of lists to actual lists
        index_counts[control] = all_indices.value_counts()  # Count occurrences of each index

    # Plot the counts for each verbosity control
    bar_width = 0.1  # Width of each bar
    unique_indices = sorted(set(idx for counts in index_counts.values() for idx in counts.index))  # Get all unique indices
    x_positions = range(len(unique_indices))  # Positions for the unique indices

    for i, (control, counts) in enumerate(index_counts.items()):
        heights = [counts.get(idx, -0.1) for idx in unique_indices]  # Get counts for each unique index, default to 0
        ax.bar(
            [x + i * bar_width for x in x_positions],  # Offset x positions for each control
            heights,
            width=bar_width,
            color=colors[i],
            alpha=0.7,
            label=control
        )

    # Set x-ticks to show the unique indices
    ax.set_xticks([x + (len(verbosity_controls) - 1) * bar_width / 2 for x in x_positions])  # Center x-ticks
    ax.set_xticklabels(unique_indices)  # Set x-tick labels to the unique indices

    # Customize the plot
    ax.set_title(f"Top 3 {metric.capitalize()} Scores")
    ax.set_xlabel("Indices of Points")
    ax.set_ylabel("Count")
    

if __name__ == "__main__":
    rouges = ["rouge2", "rougeL"]
    path = "wiki-qa-data/data/"
    file = "analysis.csv"
    
    analyse_data(path, file, rouges)
    print("Analysis completed.")

    df = pd.read_csv(path + file)
    verbosity_controls = df['verbosity_control'].unique()
    print("File read successfully. Ready for plotting...")

    metrics = {
        "f1": {
            "metrics": [
                {"rouge2 f1": "rouge2_f1"},
                {"rougeL f1": "rougeL_f1"},
                {"bert f1": "bert_f1"},
                {"similarity": "similarity"}
            ],
            "xlim": (-0.3, 1)
        },
        "length_ratio": {
            "metrics": [{"length_ratio": "length_ratio"}],
            "xlim": (0, 5)
        }
    }
    try:
        display_metric_scores(df, verbosity_controls, metrics, type="scores")
    except Exception as e:
        print(f"Error displaying metric scores: {e}")
    save_top_index_counts_to_csv(df, verbosity_controls, ["rouge2_f1", "rougeL_f1", "bert_f1", "similarity"], path+"top_index_counts.csv")
    display_metric_scores(df, verbosity_controls, metrics, type="top3")