from llm_factory import OpenAILLMs, GoogleAILLMs, DeepSeekLLMs, OllamaLLMs
import pandas as pd
import os
import time

"""

  Generate questions based on the sentences from wiki-conciseness-data.

  GOAL: 
  Get very verbose questions in order to avoid ambiguity in the answers 
  => the more verbose the question, the more likely the answer will be specific.

  Save those questions and the corresponding sentences in a file.
  THEN in llm_consistency.py: Use the questions to generate answers using the LLM model and compare their similarity with the original sentences.

"""


def fetch_data(file_path):
  """
  Fetch data from a tsv file.
  """
  data = pd.read_csv(file_path, sep="\t")
  data.columns = ["sentence", "concise_1", "concise_2", "url"] # for line we have 2 concise sentences, for full we have 5
  return data

def generate_question(llm, sentence):
  """
  Generate a question based on the provided sentence using an LLM model.
  """
  # prompt = f"Write a question based on the following sentence:\n\n{sentence}\n\nQuestion:"
  prompt = f"Write a question that can be answered directly by the following sentence:\n\n{sentence}\n\nQuestion:"
  response = llm.invoke(prompt)
  gen_question = response.content.strip()
  return gen_question, response

def avoid_rate_limit(llm_choice):
  """
  Avoid rate limit by sleeping for 60 seconds.
  """
  if llm_choice == "google":
    print("Sleeping for 60 seconds to avoid Free Tier rate limit...")
    time.sleep(60)

if __name__ == "__main__":

  file_path = "wiki-conciseness-data/concise_lite.tsv"
  output_file_path = "wiki-qa-data/concise_lite-{origin}.tsv"
  data = fetch_data(file_path)
  concise_count = data.columns.str.contains("concise").sum()

  # if output file does not exist or is empty, add the headers
  if not os.path.exists(output_file_path.format(origin="sentence")) or os.stat(output_file_path.format(origin="sentence")).st_size == 0:
    with open(output_file_path.format(origin="sentence"), "a") as f:
      f.write("sentence\tquestion\tllm_model\n")
  for c_i in range(1, concise_count+1):
    if not os.path.exists(output_file_path.format(origin=f"concise_{c_i}")) or os.stat(output_file_path.format(origin=f"concise_{c_i}")).st_size == 0:
      with open(output_file_path.format(origin=f"concise_{c_i}"), "a") as f:
        f.write("sentence\tquestion\tllm_model\n")

  # NOTE: GET ONLY TOP X ROWS
  data = data.head(20)

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
  
  # Use the original sentence from wiki to generate a question
  for (i, sentence) in enumerate(data["sentence"]):
    if i % 15 == 0 and i != 0: avoid_rate_limit(llm_choice)
    gen_question, llm_response = generate_question(llm, sentence)
    print(f"Generated Question {i+1}:", gen_question)
    with open(output_file_path.format(origin="sentence"), "a") as f:
      f.write(f"{sentence}\t{gen_question}\t{llm_model_name}\n")
  
  print("\n")
  print(f"Finished generating questions for the original sentences: {len(data)}")
  print("\n")
  avoid_rate_limit(llm_choice)

  # Use the concise sentences from wiki to generate a question
  for c_i in range(1, concise_count+1):
    for (i, sentence) in enumerate(data[f"concise_{c_i}"]):
      if i % 15 == 0 and i != 0: avoid_rate_limit(llm_choice)
      gen_question, llm_response = generate_question(llm, sentence)
      print(f"Generated Question {i+1}:", gen_question)
      with open(output_file_path.format(origin=f"concise_{c_i}"), "a") as f:
        f.write(f"{sentence}\t{gen_question}\t{llm_model_name}\n")
    print("\n")
    print(f"Finished generating questions for the concise_{c_i} sentences: {len(data)}")
    print("\n")
    if c_i != concise_count+1: avoid_rate_limit(llm_choice)