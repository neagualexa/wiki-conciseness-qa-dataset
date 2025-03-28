from llm_factory import OpenAILLMs, GoogleAILLMs, DeepSeekLLMs, OllamaLLMs
import pandas as pd
import os
import time
import re

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
    prompt = f"Using the key concepts from the URL: {url}, generate a question that is directly answered by the following sentence:{sentence}\n\nEnsure that the question is clear, concise, and directly leads to the given sentence as an answer.\nQuestion:"
  else:
    prompt = f"Generate a question that is directly answered by the following sentence:{sentence}\n\nEnsure that the question is clear, concise, and directly leads to the given sentence as an answer.\nQuestion:"

  response = llm.invoke(prompt)
  print(response)
  gen_question = response["output"] if search else response.content.strip()
  return gen_question, response

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
        return int(match.group(1))
    return 60  # Default retry delay if not found

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
        retry_delay = extract_retry_delay(str(e))
        avoid_rate_limit(llm_choice, retry_delay=60)
        gen_question, llm_response = generate_question(llm, sentence, data["url"][i], search=search)

      print(f"{origin} - Generated Question {i+1}:", gen_question)
      with open(output_file_path.format(origin=origin), "a") as f:
        f.write(f"{sentence}\t{gen_question}\t{llm_model_name}\n")
  
    print("\n")
    print(f"Finished generating questions for the {origin}: {len(data)}")
    print("\n")