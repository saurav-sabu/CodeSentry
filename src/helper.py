from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from git import Repo
import os
import re



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


def markdown_to_text(markdown_string):
    """
    Converts a markdown string to plain text with proper spacing.

    Args:
        markdown_string: The markdown string to convert.

    Returns:
        The plain text equivalent of the markdown string with improved spacing.
    """

    # Remove headers (e.g., # Header, ## Subheader, etc.)
    text = re.sub(r'#+\s*(.+)', r'\1', markdown_string)

    # Remove bold and italic formatting (e.g., **bold**, *italic*)
    text = re.sub(r'\*{1,2}(.*?)\*{1,2}', r'\1', text)
    text = re.sub(r'_{1,2}(.*?)_{1,2}', r'\1', text)

    # Remove inline code (e.g., `code`)
    text = re.sub(r'`(.*?)`', r'\1', text)

    # Remove blockquotes
    text = re.sub(r'>\s*(.*)', r'\1', text)

    # Remove links (e.g., [text](url))
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)

    # Remove images (e.g., ![alt text](url))
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

    # Remove horizontal rules
    text = re.sub(r'---', '', text)
    text = re.sub(r'___', '', text)
    text = re.sub(r'\*\*\*', '', text)

    # Remove lists (e.g., * item, 1. item)
    text = re.sub(r'(\*|\d+\.)\s+', '', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove LaTeX math (e.g., $math$, $$math$$)
    text = re.sub(r'\${1,2}(.*?)\${1,2}', r'\1', text)

    # Remove extra whitespace and newlines, preserving single newlines for paragraph breaks.
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)

    # Add extra space after periods and commas, and before newlines, for better readability.
    text = re.sub(r'([.,])(\S)', r'\1 \2', text)
    text = re.sub(r'(\S)\n', r'\1 \n', text)

    return text.strip()