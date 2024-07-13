from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
import logging
from pymongo import MongoClient
from datetime import datetime


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = MongoClient("mongodb://mongodb:27017")  # Update to use the service name from Docker Compose
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

class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model="phi3:instruct", temperature=0)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        prompt_template = """
            AI Assistant for Answering Contextual Questions
            You are an RAG chatbot to answer questions based on the provided context only.

            Instructions:
            Given the context and the question, provide a conversational answer to the user.
            Do not make up any answers. If the context does not contain the answer, simply state, "I don't know."
            Context:
            {context}

            Question:
            {question}
            Answer this question based on the provided context only. Please do not use any additional information. If you do not know, say "I don't know"
                    """


        self.prompt  = PromptTemplate(
            template = prompt_template, 
            input_variables = ["context", "question"]
        )

    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
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