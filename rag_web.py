from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
import os
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
import time
import sys
import logging
from pymongo import MongoClient
from datetime import datetime
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB client setup
client = MongoClient("mongodb://localhost:27017")
db = client["logging_db"]
collection = db["query_logs"]

def log_to_mongodb(question, answer):
    log_entry = {
        "timestamp": datetime.utcnow(),
        "question": question,
        "answer": answer
    }
    collection.insert_one(log_entry)
    logger.info(f"Logged to MongoDB: {log_entry}")

class ChatWEB:
    vector_store = None
    retriever = None
    chain = None
    link = None

    def __init__(self):
        self.model = ChatOllama(model="phi3:instruct", temperature=0)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1900, chunk_overlap=200)
        prompt_template = """
            AI Assistant Instruction:

            You are an RAG chatbot for answering questions based solely on the provided context. Given the context and the question, provide a conversational answer to the user. Do not make up any answers. If the context does not provide an answer, simply say, "I don't know."

            Context: {context}

            Question: {question}

            Answer this question from the above-given context only. Please do not use any additional information. If you do not know, say "I don't know."
                    """
        self.prompt  = PromptTemplate(
            template = prompt_template, 
            input_variables = ["context", "question"]
        )


    def ingest(self, link: str):
        self.link = link
        parsed_url = urlparse(link)
        domain_name = parsed_url.netloc.split('.')[0]
        if domain_name == 'www':
            domain_name = parsed_url.netloc.split('.')[1]
            
        data_dir = 'data'
        domain_dir = os.path.join(data_dir, domain_name)
        os.makedirs(domain_dir, exist_ok=True)
        loader=AsyncChromiumLoader([link])
        tt=Html2TextTransformer()
        docs=tt.transform_documents(loader.load())
        with open(f'{domain_dir}/data.txt', 'w', encoding='utf-8') as f:
            for doc in docs:
                f.write(doc.page_content)
        
        output_file = f"{domain_dir}/data.txt"
            
        store_path = f"vector/{domain_name}"

        # Check if vectors already exist, if not, create them
        if not os.path.exists(store_path):
            with open(output_file, 'r', encoding='utf-8') as f:
                docs = f.read()
            chunks = self.text_splitter.create_documents([docs])
            chunks = filter_complex_metadata(chunks)
            vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings(), persist_directory=store_path)
            vector_store.persist()
            
        vector_store = Chroma(persist_directory=store_path, embedding_function=FastEmbedEmbeddings())
        new_vector_store_connection = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 2
            },
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.model,
            chain_type='stuff',
            return_source_documents=False,
            retriever=new_vector_store_connection,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": self.prompt,    
                "memory": ConversationBufferMemory(
                    memory_key="history",
                    input_key="question"),
            }
        )

        self.chain = qa_chain


    def ask(self, query: str):
        if not self.chain:
            return "Please, ingest a document or a website first."
        result = self.chain({"query": query})
        answer = result["result"]
        log_to_mongodb(query, answer)
        return answer


    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None

