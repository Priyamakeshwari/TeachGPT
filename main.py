!pip install langchain
import logging
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import LlamaCpp
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from huggingface_hub import hf_hub_download
from gpt4all import GPT4All

from transformers import AutoModelForCausalLM, AutoTokenizer


from config import (
    PERSIST_DIRECTORY,
    MODEL_DIRECTORY,
    EMBEDDING_MODEL,
    DEVICE_TYPE,
    CHROMA_SETTINGS,
    MODEL_NAME,
    MODEL_FILE,
    N_GPU_LAYERS,
    MAX_TOKEN_LENGTH,
)

def load_model(model_choice, device_type=DEVICE_TYPE, model_id=MODEL_NAME, model_basename=MODEL_FILE, LOGGING=logging):
    
      
    """
    Load a language model (either LlamaCpp or GPT4All).

    Args:
        model_choice (str): The choice of the model to load ('LlamaCpp' or 'GPT4All').
        device_type (str): The type of device to use ('cuda', 'mps', or 'cpu').
        model_id (str): The ID of the model to load.
        model_basename (str): The name of the model file.
        LOGGING (logging): The logging object.

    Returns:
        LlamaCpp or GPT4All: The loaded language model.
    """
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    try:
        if model_choice == 'LlamaCpp':
            model_path = hf_hub_download(
                repo_id=model_id,
                filename=model_basename,
                resume_download=True,
                cache_dir=MODEL_DIRECTORY,
            )
            kwargs = {
                "model_path": model_path,
                "max_tokens": MAX_TOKEN_LENGTH,
                "n_ctx": MAX_TOKEN_LENGTH,
                "n_batch": 512,  
                "callback_manager": callback_manager,
                "verbose": False,
                "f16_kv": True,
                "streaming": True,
            }
            if device_type.lower() == "mps":
                kwargs["n_gpu_layers"] = 1
            if device_type.lower() == "cuda":
                kwargs["n_gpu_layers"] = N_GPU_LAYERS  # set this based on your GPU
            llm = LlamaCpp(**kwargs)
            LOGGING.info(f"Loaded {model_id} locally")
            return llm  # Returns a LlamaCpp object
        elif model_choice == 'GPT4All':
            gpt4all_model = GPT4All("orca-mini-3b.ggmlv3.q4_0.bin")
            return gpt4all_model
        elif model_choice=='HuggingFace' :
         tokenizer = AutoTokenizer.from_pretrained(model_id)
         model = AutoModelForCausalLM.from_pretrained(model_id)
         model.to(device_type)
         LOGGING.info(f"Loaded Hugging Face model '{model_id}'")
         return model
      
        else:
            LOGGING.info("Invalid model choice. Choose 'LlamaCpp' or 'GPT4All'.")
    except Exception as e:
        LOGGING.info(f"Error {e}")

def retriver(device_type=DEVICE_TYPE, LOGGING=logging):
    """
    Retrieve information using a language model and Chroma database.

    Args:
        device_type (str): The type of device to use ('cuda', 'mps', or 'cpu').
        LOGGING (logging): The logging object.
    """
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE_TYPE},
        cache_folder=MODEL_DIRECTORY,
    )
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
    )
    retriever = db.as_retriever()

    model_choice = input("Choose a model (LlamaCpp or GPT4All): ")
    
    model = load_model(model_choice, device_type, model_id=MODEL_NAME, model_basename=MODEL_FILE, LOGGING=logging)

    if model_choice == 'LlamaCpp':
        while True:
            question = input("Enter your question (type 'exit' to quit): ")
            if question.lower() == 'exit':
                break
            response = model(question)
            print(response)
    elif model_choice == 'GPT4All':
        while True:
            question = input("Enter your question (type 'exit' to quit): ")
            if question.lower() == 'exit':
                break
            response = model.generate(question, max_tokens=50)
            print(response)

if __name__ == '__main__':
    retriver()
