from flask import Flask, render_template, jsonify, request
from src.helper import download_huggingface_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app=Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embeddings=download_huggingface_embeddings()

index_name="medical-chatbox"

docSearch=PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

retriever=docSearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  
    temperature=0.4,
    google_api_key=os.environ["GEMINI_API_KEY"],
    max_output_tokens=1000
)

prompt=ChatPromptTemplate.from_messages(
  [
    ("system", system_prompt),
    ("human", "{input}"),
  ]
)


question_answer_chain=create_stuff_documents_chain(llm, prompt)
rag_chain= create_retrieval_chain(retriever, question_answer_chain)

@app.route('/')
def home():
    return "Medical Chatbot API is running."


@app.route('/api/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question is required"}), 400

    try:
        response = rag_chain.invoke({"input": question})
        answer = response.get("answer", "I don't know.")
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)