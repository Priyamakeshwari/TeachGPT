import logging
import csv
from langchain.document_loaders import DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from config import (
    PERSIST_DIRECTORY,
    MODEL_DIRECTORY,
    SOURCE_DIR,
    EMBEDDING_MODEL,
    DEVICE_TYPE,
    CHROMA_SETTINGS,
)

#TSV files loader function
def load_tsv_file(your_tsv_file.tsv):
    tsv_data = []
    with open(your_tsv_file.tsv, 'r', newline='', encoding ='utf-8') as tsv_file:
        reader = csv.DictReader(tsv_file,delimiter='\t')
        
        for row in reader:
            tsv_data.append(row)
    return tsv_data
    
    
def load_docs(directory: str = SOURCE_DIR):
    loader = DirectoryLoader(directory, glob="**/*.pdf", use_multithreading=True, loader_cls=PDFMinerLoader)
    docs = loader.load()
    logging.info(f"Loaded {len(docs)} documents from {directory}")
    return docs

def split_docs(documents,chunk_size=1000,chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    logging.info(f"Split {len(docs)} documents into chunks")
    return docs

def builder():
    logging.info("Building the database")
    documents = load_docs()
    docs = split_docs(documents)
    
     # Load TSV file using TSV loader function
    tsv_data = load_tsv_file('your_tsv_file.tsv')

    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE_TYPE},
        cache_folder=MODEL_DIRECTORY,
    )
    db = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
        tsv_data=tsv_data  # Pass the loaded TSV data here

    )
    logging.info(f"Loaded Documents to Chroma DB Successfully")


if __name__ == '__main__':
 
    builder()