#!/usr/bin/env python
import os

from colorama import Fore, Style
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import Chroma
from openai import RateLimitError


def discover_documents(dir: str = "documents"):
    """Find all docs in the specified folder"""
    docs = [
        os.path.join(dir, f)
        for f in os.listdir(dir)
        if os.path.isfile(os.path.join(dir, f))
    ]
    return docs


def load_pdf(file: str):
    """Reads a PDF document and returns a text version of each page"""
    print("Reading", file)
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    return pages


def read_all_docs(dir: str = "documents"):
    """Read and parse all documents"""
    res = []
    for d in discover_documents(dir):
        res.extend(load_pdf(d))
    return res


def setup():
    """Get the model ready to answer questions"""
    docs = read_all_docs()
    while True:
        try:
            vector_store = Chroma.from_documents(
                documents=docs, embedding=OpenAIEmbeddings()
            )
            break
        except RateLimitError as e:
            print("Too many tokens:", e["error"]["message"])
            # Chuck away half
            docs = docs[0 : len(docs) // 2]

    retriever = vector_store.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return docs, rag_chain


def format_bot(x: str):
    """Pretty printing"""
    txt = Fore.BLUE + "Bot: " + Style.RESET_ALL + x + Fore.RESET
    return txt


def format_user(x: str = ""):
    """Pretty printinge"""
    txt = Fore.GREEN + "User: " + Style.RESET_ALL + x + Fore.RESET
    return txt


if __name__ == "__main__":
    print("Training the robot, please wait...")
    docs, rag_chain = setup()
    print("done!")

    print("Type your question at the prompt and press ENTER, or 'Q' to quit.")

    while True:
        query = input(format_user()).strip()

        if query.lower() == "q":
            break

        print(format_bot(rag_chain.invoke(query)))

    print(format_bot("Righto, see you later!"))
