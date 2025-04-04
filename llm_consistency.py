from llm_factory import OpenAILLMs, GoogleAILLMs, DeepSeekLLMs, OllamaLLMs
import pandas as pd
import os
import re
import time
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
from nltk.util import ngrams
from collections import Counter
import nltk
import bert_score
import verbosity_control_prompts
# nltk.download('punkt')

"""

  Answer the questions generated in questions_gen.py using the LLM model and compare their similarity with the original sentences.

  GOAL: 
  Observe the verbosity of the answers by comparing their similarity with the original sentences.

  NOTE: answering LLM should not have access to any external tools as it introduces a new variable that can affect the results.

"""


def fetch_data(file_path, tsv=False, json=False):
  """
  Fetch data from a tsv file.
  """
  if tsv:
    data = pd.read_csv(file_path, sep="\t")
  elif json:
    data = pd.read_json(file_path, lines=True)
  else:
    raise ValueError("Invalid file format")

  return data

def generate_answer(llm, verbosity_control, question):
  """
  Based on the verbosity control prompt, invoke the LLM model to answer the question.
  """
  prompt = f"{verbosity_control}\n\nQuestion:{question}\n\nAnswer:"
  response = llm.invoke(prompt)
  gen_answer = " ".join(response.content.strip().split("\n"))
  return gen_answer, response

def get_cosine_similarity(sentence_transformer, sentence, answer):
  embeddings = sentence_transformer.encode([sentence, answer], convert_to_tensor=True)
  similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
  return similarity

def calculate_length_ratio(generated, ground_truth):
    """
    Source: "Precise Length Control in Large Language Models" (Butcher et al., 2024)
    """
    return len(generated.split()) / len(ground_truth.split())

def calculate_compression_rate(generated, ground_truth):
    """
    Source: "Precise Length Control in Large Language Models" (Butcher et al., 2024)
    """
    return 1 - (len(ground_truth.split()) / len(generated.split()))

def calculate_rouge_score(generated, ground_truth, rouges):
    """
    Source: "Conciseness: An Overlooked Language Task" (Stahlberg et al., 2022)

    Calculate ROUGE scores for the generated answer and the ground truth.
      ROUGE 1: Measure overap of unigrams (single words) - basic content matching
      ROUGE 2: Measure overlap of bigrams - content matching with some sentence structure
      ROUGE L: Measure longest common subsequence of words - content matching with sentence structure
      # ROUGE W: Measure weighted longest common subsequence of words - content matching with sentence structure
      # ROUGE S: Measure skip-bigram (words in the pair can have a gap between them) - content matching with sentence structure
      # ROUGE SU: Measure skip-bigram and unigram - content matching with sentence structure
    """
    scorer = rouge_scorer.RougeScorer(rouges, use_stemmer=True)
    scores = scorer.score(target=ground_truth, prediction=generated)
    return scores

def calculate_redundancy_score(text, n=2):
    """
    Source: "Verbosity ≠ Veracity: Demystify Verbosity Compensation Behavior of Large Language Models" (Zhang et al., 2024)
    """
    words = text.split()
    n_grams = list(ngrams(words, n))
    n_gram_counts = Counter(n_grams)
    total_n_grams = len(n_grams)
    repeated_n_grams = sum(count - 1 for count in n_gram_counts.values() if count > 1)
    return repeated_n_grams / total_n_grams if total_n_grams > 0 else 0

def calculate_BERTScore(generated, ground_truth):
    """
    Source: "BERTScore: Evaluating Text Generation with BERT" (Zhang et al., 2024)
    """
    P, R, F1 = bert_score.score([generated], [ground_truth], lang='en', verbose=False, rescale_with_baseline=True)
    return {
        "precision": P.item(),
        "recall": R.item(),
        "f1": F1.item()
    }

def avoid_rate_limit(llm_choice, retry_delay=60):
  """
  Avoid rate limit by sleeping for retry_delay seconds.
  """
  if llm_choice == "google":
    print(f"Sleeping for {retry_delay} seconds to avoid Free Tier rate limit...")
    time.sleep(retry_delay)

def extract_retry_delay(error_message):
    """
    Extract the retry delay in seconds from the error message.
    """
    match = re.search(r"retry_delay\s*{\s*seconds:\s*(\d+)", error_message)
    if match:
        return int(match.group(1)) + 10  # Add 10 seconds to avoid rate limit
    return 60  # Default retry delay if not found

# --------------------

def generate_LLM_responses(path, file_path, output_file_path, llm, llm_model_name, verbosity_controls):
  """
  Generate LLM responses for the questions.
  """

  # for all files in the path directory
  for file in os.listdir(path):
    if file.endswith(".tsv") and "-ans" not in file:
      origin = file.split('.')[0].split("-")[1]
      data = fetch_data(file_path.format(origin=origin), tsv=True)

      if not os.path.exists(output_file_path.format(origin=origin)) or os.stat(output_file_path.format(origin=origin)).st_size == 0:
        with open(output_file_path.format(origin=origin), "a") as f:
          f.write("question\tgold_sentence\tanswer\tverbosity_control\tllm_model_question\tllm_model_answer\n")
  
      # NOTE: GET ONLY TOP X ROWS
      # data = data.head(10)
      
      # Use the original sentence from wiki to generate a question
      for (i, question) in enumerate(data["question"]):
        for (i_v, verbosity_control) in enumerate(verbosity_controls):
          loop_count = i * len(verbosity_controls) + i_v
          try:
            gen_answer, llm_response = generate_answer(llm, verbosity_control, question)
          except Exception as e:
            print(f"Error: {e}")
            retry_delay = extract_retry_delay(str(e))
            avoid_rate_limit(llm_choice, retry_delay=60)
            gen_answer, llm_response = generate_answer(llm, verbosity_control, question)

          print(f"{origin}-{loop_count+1} - Generated Answer {i+1} with Verbosity Control {i_v+1}: {verbosity_control}")

          with open(output_file_path.format(origin=origin), "a") as f:
            f.write(f"{question}\t{data['sentence'][i]}\t{gen_answer}\t{verbosity_control}\t{data['llm_model'][i]}\t{llm_model_name}\n")
      
      print("\n")
      print(f"Finished generating questions for the original sentences: {len(data)} with verbosity control: {verbosity_control}")
      print("\n")

def generate_LLM_responses_clapnq(file_path, output_file_path, llm, llm_model_name, verbosity_controls, data_limit=10):
  """
  Generate LLM responses for the questions in CLAPNQ dataset.
  """
  data = fetch_data(file_path, json=True)
  seed = 4321
  data = data.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle 

  data = data.head(data_limit)

  if not os.path.exists(output_file_path) or os.stat(output_file_path).st_size == 0:
      with open(output_file_path, "a") as f:
        f.write("question\tgold_sentence\tanswer\tverbosity_control\tllm_model_question\tllm_model_answer\tcontext\n")

  for di, datapoint in data.iterrows():
      question = datapoint["input"].replace("\n", " ")
      context = str(datapoint["passages"]).replace("\n", " ")
      ground_truth = datapoint["output"][0]["answer"].replace("\n", " ").replace("\t", " ").replace("\r", " ") # only get the first human annotated answer
      for (i_v, verbosity_control) in enumerate(verbosity_controls):
          loop_count = di * len(verbosity_controls) + i_v

          # Concatenate Context
          verbosity_control_prompt = f"Context:{context}\n\n{verbosity_control}"

          try:
            gen_answer, llm_response = generate_answer(llm, verbosity_control_prompt, question)
          except Exception as e:
            print(f"Error: {e}")
            retry_delay = extract_retry_delay(str(e))
            avoid_rate_limit(llm_choice, retry_delay=60)
            gen_answer, llm_response = generate_answer(llm, verbosity_control_prompt, question)

          print(f"clapnq-{loop_count+1} - Generated Answer {di+1} with Verbosity Control {i_v+1}: {verbosity_control}")

          with open(output_file_path, "a") as f:
            f.write(f"{question}\t{ground_truth}\t{gen_answer}\t{verbosity_control}\thuman\t{llm_model_name}\t{context}\n")

def generate_LLM_responses_mathdial(file_path, output_file_path, llm, llm_model_name, verbosity_controls, data_limit=10):
  """
  Generate LLM responses for the questions in CLAPNQ dataset.
  """
  data = fetch_data(file_path, json=True)
  seed = 12
  
  data["conversation_list"] = [c.split('|EOM|') for c in data["conversation"].tolist()]
  data = data[(data["conversation_list"].str.len() > 3) & \
              (data["conversation_list"].str[2].str.contains("Teacher:")) & (data["conversation_list"].str[1].str.contains("Student:"))]

  data = data.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle 

  data = data.head(data_limit)

  if not os.path.exists(output_file_path) or os.stat(output_file_path).st_size == 0:
      with open(output_file_path, "a") as f:
        f.write("question\tgold_sentence\tanswer\tverbosity_control\tllm_model_question\tllm_model_answer\tcontext\n")

  for di, datapoint in data.iterrows():
      question = "Help me find what I am wrong on."
      context = "Exercise Question: " + datapoint["question"] + \
                "\n\nSolution: " + datapoint["ground_truth"] + \
                "\n\nStudent's Submission on the exercise: " + datapoint["student_incorrect_solution"] + \
                "\n\nConversation until now: " + " ".join(datapoint["conversation"].split('|EOM|')[:2])
                # "\n\nStudent's confusion: " + datapoint["teacher_described_confusion"] + \
      ground_truth = datapoint["conversation"].split('|EOM|')[2].replace("\n", " ").replace("\t", " ").replace("\r", " ")\
                      .split('Teacher: ')[1]
      # dataset has a whole groundtruth conversation, only look at the first teacher response
      for (i_v, verbosity_control) in enumerate(verbosity_controls):
          loop_count = di * len(verbosity_controls) + i_v

          # Concatenate Context
          verbosity_control_prompt = f"{verbosity_control}\n\n{context}"

          try:
            gen_answer, llm_response = generate_answer(llm, verbosity_control_prompt, question)
          except Exception as e:
            print(f"Error: {e}")
            retry_delay = extract_retry_delay(str(e))
            avoid_rate_limit(llm_choice, retry_delay=60)
            gen_answer, llm_response = generate_answer(llm, verbosity_control_prompt, question)

          print(f"mathdial-{loop_count+1} - Generated Answer {di+1} (length {len(gen_answer)}) with Verbosity Control {i_v+1}")

          with open(output_file_path, "a") as f:
            sanitized_context = re.sub(r"[\n\t\r]", " ", context)
            sanitized_verbosity_control = re.sub(r"[\n\t\r]", " ", verbosity_control)
            f.write(f"{question}\t{ground_truth}\t{gen_answer}\t{sanitized_verbosity_control}\thuman\t{llm_model_name}\t{sanitized_context}\n")

# --------------------

"""
Calculate similarity between the original sentence and the generated answer.

NOTE: develop a metric based on this:
- if llm response is shorter than ground truth and is contained in the ground truth, then great
- if llm response is around the same length as ground truth and is contained in the ground truth, then good
- if llm response is longer than ground truth and contains the ground truth, then good but verbose
- if llm response does not contain the ground truth regardless of length, then bad
also look at redundancy in the answers
"""
def calculate_similarity(generated, ground_truth, sentence_transformer=None):
    """
    Calculate similarity between the original sentence and the generated answer.
    """
    # Length-based Metrics
    length_ratio = calculate_length_ratio(generated, ground_truth)
    compression_rate = calculate_compression_rate(generated, ground_truth)
    
    # ROUGE Scores
    rouges = ["rouge2", "rougeL"]
    rouge_scores = calculate_rouge_score(generated, ground_truth, rouges=rouges)
    rouge_2_f1 = rouge_scores["rouge2"].fmeasure
    rouge_L_f1 = rouge_scores["rougeL"].fmeasure
    rouge_scores = {rouge: { "f1": rouge_scores[rouge].fmeasure, \
                            "precision": rouge_scores[rouge].precision, \
                            "recall": rouge_scores[rouge].recall} \
                            for rouge in rouges}
    
    # BERTScore
    bert_scores = calculate_BERTScore(generated, ground_truth)
    
    # # Semantic Similarity
    # if sentence_transformer is None:
    #     sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')  # Load model if not provided
    # embeddings = sentence_transformer.encode([ground_truth, generated], convert_to_tensor=True)
    # semantic_similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    
    # Weighted Combination of Metrics
    similarity_score = (
        (0.5 * bert_scores["f1"]) + # Semantic similarity for meaning
        (0.4 * rouge_L_f1) +  # ROUGE-L for structural similarity
        (0.1 * (1 - length_ratio))  # Inverse length ratio for conciseness
    )
    
    return max(0, min(1, similarity_score)), rouge_scores, bert_scores, length_ratio # Ensure score is between 0 and 1


# --------------------

def analyse_LLM_consistency(path):
    """
    Analyze the consistency of LLM responses by calculating similarity scores
    """
    # sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

    for file in os.listdir(path):
        if file.endswith(".tsv") and "-ans" in file:
            file_path = os.path.join(path, file)
            data = pd.read_csv(file_path, sep="\t")

            # Delete existing similarity columns: similarity	rouge_scores	cosine_similarity	bert_scores	length_ratio
            if all(col in data.columns for col in ["similarity", "rouge_scores", "bert_scores", "length_ratio"]):
              data.drop(columns=['similarity', 'rouge_scores', 'bert_scores', 'length_ratio'], inplace=True)

            # Add a new column for similarity scores
            similarity_scores = []
            rouge_scores = []
            bert_scores = []
            length_ratios = []

            # Iterate through the rows and calculate similarity
            for i in range(len(data)):
                similarity, rouge_score, bert_score, length_ratio = calculate_similarity(data["answer"][i], data["gold_sentence"][i])
                rouge_scores.append(rouge_score)
                bert_scores.append(bert_score)
                length_ratios.append(length_ratio)
                similarity_scores.append(similarity)

            # Add the similarity scores to the DataFrame
            data["length_ratio"] = length_ratios
            data["rouge_scores"] = rouge_scores
            data["bert_scores"] = bert_scores
            data["similarity"] = similarity_scores

            # Save the updated DataFrame back to the CSV file
            data.to_csv(file_path, sep="\t", index=False)
            print(f"Updated file: {file_path} with similarity scores.\n\n")

if __name__ == "__main__":
    llm_choice = "google" # or "google", "deepseek", "ollama"
    if llm_choice == "openai":
      llm = OpenAILLMs().get_llm()
      llm_model_name = OpenAILLMs().get_model_name()
    elif llm_choice == "google":
      llm = GoogleAILLMs().get_llm()
      llm_model_name = GoogleAILLMs().get_model_name()
    elif llm_choice == "deepseek":
      llm = DeepSeekLLMs().get_llm()
      llm_model_name = DeepSeekLLMs().get_model_name()
    elif llm_choice == "ollama":
      llm = OllamaLLMs().get_llm()
      llm_model_name = OllamaLLMs().get_model_name()
    else:
      raise ValueError("Invalid LLM choice")
    
    # verbosity_controls = [
    #   "Directly answer the student's question. Keep your answer short.",
    #   "Answer the student's question. Keep your answer short.",
    #   "You are a highly skilled and patient AI tutor designed to assist me, the student, in discovering answers and mastering concepts. Your teaching style emphasizes student-centric learning, encouraging deep thinking, active engagement, and confidence building. Keep your answer short.",
    #   "You are a highly skilled and patient AI tutor. Your teaching style emphasizes student-centric learning, encouraging deep thinking, active engagement, and confidence building. Keep your answer short."
    # ]
    verbosity_controls = verbosity_control_prompts.verbosity_controls

    # CONCISENESS DATASET
    # path = "wiki-qa-data/"
    # output_path = "wiki-qa-data/"
    # file_path = path+"concise_lite-{origin}.tsv"
    # output_file_path = path+"concise_lite-{origin}-ans.tsv"
    # generate_LLM_responses(path, file_path, output_file_path, llm, llm_model_name, verbosity_controls)

    # CLAPNQ DATASET
    # path = "clapnq-data/"
    # output_path = "wiki-qa-data/"
    # file_path = path+"clapnq_dev_answerable.jsonl"
    # output_file_path = output_path+"clapnq_dev_answerable-ans.tsv"
    # generate_LLM_responses_clapnq(file_path, output_file_path, llm, llm_model_name, verbosity_controls, data_limit=20)

    # MATHDIAL DATASET
    path = "mathdial-data/"
    output_path = "wiki-qa-data/"
    file_path = path+"train.jsonl"
    output_file_path = output_path+"mathdial-train-ans.tsv"
    generate_LLM_responses_mathdial(file_path, output_file_path, llm, llm_model_name, verbosity_controls, data_limit=30)

    print("LLM responses generated.")
    analyse_LLM_consistency(output_path)
    print("Analysis completed.")