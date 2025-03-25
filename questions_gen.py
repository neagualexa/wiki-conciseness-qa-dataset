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

def generate_question(llm, sentence, url, search=False):
  """
  Generate a question based on the provided sentence using an LLM model.
  """
  if search:
    prompt = f"Based on the key concepts from this URL: {url}, write a question that can be answered directly by the following sentence:\n\n{sentence}\n\nQuestion:"
  else:
    prompt = f"Write a question that can be answered directly by the following sentence:\n\n{sentence}\n\nQuestion:"

  response = llm.invoke(prompt)
  print(response)
  gen_question = response["output"] if search else response.content.strip()
  return gen_question, response

def avoid_rate_limit(llm_choice):
  """
  Avoid rate limit by sleeping for 60 seconds.
  """
  if llm_choice == "google":
    print("Sleeping for 60 seconds to avoid Free Tier rate limit...")
    time.sleep(60)

if __name__ == "__main__":

  search = True # Allow Search to wiki url for more context
  file_path = "wiki-conciseness-data/concise_lite.tsv"
  output_file_path = "wiki-qa-data/concise_lite-{origin}.tsv"
  data = fetch_data(file_path)
  concise_count = data.columns.str.contains("concise").sum()

  origins = ["sentence"] + [f"concise_{i}" for i in range(1, concise_count+1)]

  # NOTE: GET ONLY TOP X ROWS
  data = data.head(20)

  llm_choice = "google" # or "google", "deepseek", "ollama"
  if llm_choice == "openai":
    llm = OpenAILLMs().get_llm()
    llm_model_name = OpenAILLMs().get_model_name()
  elif llm_choice == "google":
    llm = GoogleAILLMs(search=search).get_llm(search=search)
    llm_model_name = GoogleAILLMs().get_model_name()
  elif llm_choice == "deepseek":
    llm = DeepSeekLLMs().get_llm()
    llm_model_name = DeepSeekLLMs().get_model_name()
  elif llm_choice == "ollama":
    llm = OllamaLLMs().get_llm()
    llm_model_name = OllamaLLMs().get_model_name()
  else:
    raise ValueError("Invalid LLM choice")
  
  for origin in origins:
    if origin not in data.columns:
      raise ValueError(f"Data must contain '{origin}' column.")
    if not os.path.exists(output_file_path.format(origin=origin)) or os.stat(output_file_path.format(origin=origin)).st_size == 0:
      with open(output_file_path.format(origin=origin), "a") as f:
        f.write("sentence\tquestion\tllm_model\n")

    for (i, sentence) in enumerate(data[origin]):
      try:
        gen_question, llm_response = generate_question(llm, sentence, data["url"][i], search=search)
      except Exception as e:
        print(f"Error: {e}")
        avoid_rate_limit(llm_choice)
        gen_question, llm_response = generate_question(llm, sentence, data["url"][i], search=search)

      print(f"{origin} - Generated Question {i+1}:", gen_question)
      with open(output_file_path.format(origin=origin), "a") as f:
        f.write(f"{sentence}\t{gen_question}\t{llm_model_name}\n")
  
    print("\n")
    print(f"Finished generating questions for the {origin}: {len(data)}")
    print("\n")