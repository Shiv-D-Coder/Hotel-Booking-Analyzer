# api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
import chromadb

# Initialize FastAPI app
app = FastAPI()

# Set up Groq API Key (replace with your actual key)
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"

# Initialize ChromaDB client and collection
client = chromadb.Client()
collection_name = "hotel_bookings"
vectorstore = Chroma(client=client, collection_name=collection_name)

# Initialize Groq LLM (Llama-3.3-70B-Versatile)
llm = ChatGroq(model="llama-3.3-70b-versatile")

# Create a RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Pydantic model for analytics endpoint input
class AnalyticsRequest(BaseModel):
    year: int
    month: str

# Pydantic model for Q&A endpoint input
class QuestionRequest(BaseModel):
    question: str

# POST /analytics → Returns analytics reports.
@app.post("/analytics")
async def get_analytics(request: AnalyticsRequest):
    try:
        # Example analytics logic (you can customize this based on your dataset)
        year = request.year
        month = request.month

        # Mocked response for demonstration purposes; replace with actual logic.
        response = f"Analytics report for {month} {year}: Total revenue is €XXX, cancellations are YYY."

        return {"status": "success", "data": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# POST /ask → Answers booking-related questions.
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        question = request.question

        # Use the RAG pipeline to answer the question
        result = qa_chain.run(question)

        return {"status": "success", "answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint for health check
@app.get("/")
async def root():
    return {"message": "RAG API is running successfully!"}
