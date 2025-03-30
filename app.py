from src.helper import *
from src.prompt import *
from langchain_google_genai import  ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from git import Repo
from dotenv import load_dotenv
from flask import Flask,render_template,jsonify,request
from pathlib import Path

app = Flask(__name__)

embedding = initialize_embedding()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chatbot",methods=["GET","POST"])
def getRepo():
    if request.method=="POST":
        user_input = request.form["repo_link"]
        create_repo(user_input)
        os.system("python store_index.py")
        

    return jsonify({"response":str(user_input)})

@app.route("/get",methods=["GET","POST"])
def chat():
    msg = request.form["msg"]
    input = msg

    if input == "clear":
        os.system("rm -rf repo")
    
    chat_history = []
    vector_db = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever()

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

    history_aware_retriever = create_history_aware_retriever(
                llm, retriever, contextualize_q_prompt
            )

    prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

    question_answer_chain = create_stuff_documents_chain(llm,prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)
    result = rag_chain.invoke({"input": input,"chat_history":chat_history})
    return str(result["answer"])

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080,debug=True)