from src.helper import *
from langchain_community.vectorstores import FAISS

print("executing store_index.py")
documents = load_data("repo/")
chunks = text_splitter(documents)
embedding = initialize_embedding()

vector_db = FAISS.from_documents(chunks,embedding)
vector_db.save_local("faiss_index")

print("executed store_index.py")
