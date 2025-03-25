from llm_factory import OpenAILLMs, GoogleAILLMs, DeepSeekLLMs, OllamaLLMs
import pandas as pd
import os
import time
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
from nltk.util import ngrams
from collections import Counter
import nltk
# nltk.download('punkt')

"""

  Answer the questions generated in questions_gen.py using the LLM model and compare their similarity with the original sentences.

  GOAL: 
  Observe the verbosity of the answers by comparing their similarity with the original sentences.

  NOTE: develop a metric based on this:
  - if llm response is shorter than ground truth and is contained in the ground truth, then great
  - if llm response is around the same length as ground truth and is contained in the ground truth, then good
  - if llm response is longer than ground truth and contains the ground truth, then good but verbose
  - if llm response does not contain the ground truth regardless of length, then bad
  also look at redundancy in the answers

  NOTE: answering LLM should not have access to any external tools as it introduces a new variable that can affect the results.

"""


def fetch_data(file_path):
  """
  Fetch data from a tsv file.
  """
  data = pd.read_csv(file_path, sep="\t")
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

def calculate_rouge_score(generated, ground_truth):
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
    scorer = rouge_scorer.RougeScorer(["rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(ground_truth, generated)
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

def avoid_rate_limit(llm_choice):
  """
  Avoid rate limit by sleeping for 60 seconds.
  """
  if llm_choice == "google":
    print("Sleeping for 60 seconds to avoid Free Tier rate limit...")
    time.sleep(60)

if __name__ == "__main__":

  path = "wiki-qa-data/"
  file_path = path+"concise_lite-{origin}.tsv"
  output_file_path = path+"concise_lite-{origin}-ans.tsv"
  concise_count = os.path.basename(path).count("concise")

  sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2') 

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
  
  verbosity_controls = [
    "Answer the following question.",
    "Answer the following question with a concise answer.",
    "Directly answer the following question.",
    "Answer the following question without unnecessary details.",
  ]

  # for all files in the path directory
  for file in os.listdir(path):
    if file.endswith(".tsv") and "-ans" not in file:
      origin = file.split('.')[0].split("-")[1]
      data = fetch_data(file_path.format(origin=origin))

      if not os.path.exists(output_file_path.format(origin=origin)) or os.stat(output_file_path.format(origin=origin)).st_size == 0:
        with open(output_file_path.format(origin=origin), "a") as f:
          f.write("question\tgold_sentence\tanswer\tsimilarity\tverbosity_control\tllm_model_question\tllm_model_answer\n")
  
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
            avoid_rate_limit(llm_choice)
            gen_answer, llm_response = generate_answer(llm, verbosity_control, question)

          print(f"{origin}-{loop_count+1} - Generated Answer {i+1} with Verbosity Control {i_v+1}: {verbosity_control}")

          # Get similarity between the original sentence and the generated answer
          similarity = calculate_rouge_score(data["sentence"][i], gen_answer)

          with open(output_file_path.format(origin=origin), "a") as f:
            f.write(f"{question}\t{data['sentence'][i]}\t{gen_answer}\t{similarity}\t{verbosity_control}\t{data['llm_model'][i]}\t{llm_model_name}\n")
      
      print("\n")
      print(f"Finished generating questions for the original sentences: {len(data)} with verbosity control: {verbosity_control}")
      print("\n")