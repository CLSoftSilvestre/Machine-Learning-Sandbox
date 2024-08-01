# -*- coding: utf-8 -*-
"""
Created on 09/07/2024

@author: CSilvestre

ISSUES: Answer takes to long to appear...

"""
import os
from flask import Flask, render_template, send_from_directory, request, redirect, url_for, jsonify, session, send_file
from flask_session import Session
import json

from langchain_community.document_loaders import UnstructuredPDFLoader

from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

app = Flask(__name__, instance_relative_config=True)

ingestionFilePath = os.path.join(app.root_path, 'ingest', 'Test.pdf')

if (ingestionFilePath):
    loader = UnstructuredPDFLoader(file_path=ingestionFilePath)
    data = loader.load()

#Vector Embeddings

#Split and chunk
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

#Add data to vector database
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
    collection_name="local-rag"
)

#Retrieve (ask question and get response...)
local_model = "gemma:2b"
llm = ChatOllama(model=local_model)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are and AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve the relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limkitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(),
    llm,
    prompt=QUERY_PROMPT
)

# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


@app.route("/chat/", methods=['GET', 'POST'])
def chat():
    body = json.loads(request.data)
    question = body["prompt"]
    #chain.invoke(question)
    return chain.invoke(question)


@app.route("/index")
@app.route("/")
def index():
    return "API is running", 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5002, debug=True)