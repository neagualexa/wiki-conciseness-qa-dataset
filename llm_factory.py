import os

from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

class AzureLLMs:
    def __init__(self, temperature: int = 0):
        self._azure_llm = AzureChatOpenAI(
                        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
                        temperature=temperature,
                        max_tokens=None,
                    )
        self._azure_embedding = AzureOpenAIEmbeddings(azure_deployment=os.environ['AZURE_OPENAI_EMBEDDING_1536_DEPLOYMENT'], 
                                        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                                        model=os.environ["AZURE_OPENAI_EMBEDDING_1536_MODEL"])

    def get_llm(self):    
        return self._azure_llm

    def get_embedding(self):
        return self._azure_embedding
    
    def get_model_name(self):
        return os.environ['AZURE_OPENAI_CHAT_DEPLOYMENT_NAME']

class OllamaLLMs:
    def __init__(self):
        self._ollama_llm = Ollama(
            model=os.environ['OLLAMA_MODEL'], # Any of the available models listed in the API docs
            base_url=os.environ['OLLAMA_BASE_URL'],
            headers={
                'X-API-Key': os.environ['OLLAMA_API_KEY'],
            },
        )

        self._ollama_embedding = OllamaEmbeddings(
            model='nomic-embed-text:137m-v1.5-fp16',
            base_url=os.environ['OLLAMA_BASE_URL'],
            headers={
                'X-API-Key': os.environ['OLLAMA_API_KEY'],
            },
            show_progress=True
        )

    def get_llm(self):
        return self._ollama_llm

    def get_embedding(self):
        return self._ollama_embedding
    
    def get_model_name(self):
        return os.environ['OLLAMA_MODEL']
    
class OpenAILLMs:
    def __init__(self, temperature: int = 0):
        self._openai_llm = ChatOpenAI(
            model=os.environ['OPENAI_MODEL'],
            temperature=temperature,
            api_key=os.environ["OPENAI_API_KEY"],
        )

        self._openai_embedding = OpenAIEmbeddings(
            model='text-embedding-ada-002',
            api_key=os.environ['OPENAI_API_KEY'],
        )
    
    def get_llm(self):
        return self._openai_llm
    
    def get_embedding(self):
        return self._openai_embedding
    
    def get_model_name(self):
        return os.environ['OPENAI_MODEL']

class GoogleAILLMs:
    def __init__(self, temperature: int = 0):

        self._google_llm = ChatGoogleGenerativeAI(
            model=os.environ['GOOGLE_AI_MODEL'],
            temperature=temperature,
            google_api_key=os.environ['GOOGLE_AI_API_KEY'],
        )
    
    def get_llm(self):
        return self._google_llm
    
    def get_model_name(self):
        return os.environ['GOOGLE_AI_MODEL']
    
class DeepSeekLLMs:
    def __init__(self, temperature: int = 0):
        self._deepseek_llm = ChatOpenAI(
            model=os.environ['DEEPSEEK_MODEL'],
            api_key=os.environ['DEEPSEEK_API_KEY'],
            base_url='https://api.deepseek.com',
            temperature=temperature,
        )

    def get_llm(self):
        return self._deepseek_llm
    
    def get_model_name(self):
        return os.environ['DEEPSEEK_MODEL']