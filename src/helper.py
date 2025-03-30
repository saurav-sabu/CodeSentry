from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from git import Repo
import os


def create_repo(repo_url):
    os.makedirs("repo",exist_ok=True)
    Repo.clone_from(repo_url,to_path="repo/")


def load_data(repo_path):
    loader = GenericLoader.from_filesystem(
    "repo/",
    glob="**/*",
    suffixes=[".py"],
    parser=LanguageParser(language=Language.PYTHON,parser_threshold=500)
    )
    documents = loader.load()
    return documents

def text_splitter(documents):
    splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=500,
    chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    return chunks

def initialize_embedding():
    embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return embedding