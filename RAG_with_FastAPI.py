import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import faiss
from sklearn.preprocessing import normalize
from groq import Groq
import time
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from datetime import datetime
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Define API request/response models
class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class AnalyticsRequest(BaseModel):
    report_type: Optional[str] = "summary"

class RAGSystem:
    def __init__(self, csv_path: str, embedding_dim: int = 384):
        """Initialize the RAG system with a CSV file and set up FAISS"""
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.embedding_dim = embedding_dim
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(embedding_dim)
        
        # Initialize Groq client
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"
        
        # Performance tracking
        self.query_times = []
        self.retrieval_times = []
        
        # Process data
        self.process_data()
    
    def process_data(self):
        """Process the CSV data and create embeddings"""
        logger.info(f"Processing data from {self.csv_path}")
        
        # Convert columns to string
        for col in self.df.columns:
            self.df[col] = self.df[col].astype(str)
        
        # Create text chunks
        self.chunks = []
        for idx, row in self.df.iterrows():
            text = " ".join([f"{col}: {val}" for col, val in row.items()])
            self.chunks.append({
                "id": idx,
                "text": text,
                "row": row.to_dict()
            })
        
        # Generate embeddings
        self.generate_embeddings()
    
    def generate_embeddings(self):
        """Generate embeddings for all chunks"""
        logger.info("Generating embeddings...")
        
        embeddings = []
        for chunk in self.chunks:
            embedding = self.get_embedding(chunk["text"])
            embeddings.append(embedding)
        
        # Process embeddings
        self.embeddings = np.array(embeddings).astype('float32')
        self.embeddings = normalize(self.embeddings)
        
        # Add to FAISS index
        self.index.add(self.embeddings)
        logger.info(f"Added {len(self.embeddings)} embeddings to FAISS index")
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text (simplified version)"""
        # Simple deterministic embedding for demonstration
        np.random.seed(sum(ord(c) for c in text))
        embedding = np.random.normal(0, 1, self.embedding_dim)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant chunks based on a query"""
        start_time = time.time()
        
        # Generate embedding for query
        query_embedding = np.array([self.get_embedding(query)]).astype('float32')
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({
                    "chunk": self.chunks[idx],
                    "score": float(1.0 / (1.0 + distances[0][i]))
                })
        
        retrieval_time = time.time() - start_time
        self.retrieval_times.append(retrieval_time)
        
        return results
    
    def query(self, user_query: str, top_k: int = 5) -> Dict[str, Any]:
        """Process a user query using RAG"""
        start_time = time.time()
        
        # Retrieve relevant chunks
        relevant_chunks = self.search(user_query, top_k)
        
        # Prepare context
        context = ""
        for item in relevant_chunks:
            context += f"{item['chunk']['text']}\n\n"
        
        # Generate response using Groq
        prompt = f"""
        You are a helpful assistant that answers questions about hotel bookings data.
        
        Here is some context information to help you answer:
        {context}
        
        User question: {user_query}
        
        Provide a concise and accurate response based on the context provided.
        """
        
        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions about hotel bookings data."},
                {"role": "user", "content": prompt}
            ],
            model=self.model
        )
        
        total_time = time.time() - start_time
        self.query_times.append(total_time)
        
        return {
            "answer": response.choices[0].message.content,
            "context": [item["chunk"]["text"] for item in relevant_chunks[:2]],  # Return only top 2 contexts
            "performance": {
                "total_time": round(total_time, 3),
                "retrieval_time": round(self.retrieval_times[-1], 3) if self.retrieval_times else None
            }
        }
    
    def generate_analytics(self) -> Dict[str, Any]:
        """Generate key analytics metrics from hotel booking data"""
        df = self.df.copy()
        
        # Convert necessary columns to numeric
        try:
            for col in ['is_canceled', 'lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adr']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate 6-8 key metrics
            total_bookings = len(df)
            canceled_rate = df[df['is_canceled'] == 1].shape[0] / total_bookings * 100 if total_bookings > 0 else 0
            avg_lead_time = df['lead_time'].mean()
            avg_stay = (df['stays_in_weekend_nights'] + df['stays_in_week_nights']).mean()
            avg_weekend_stay = df['stays_in_weekend_nights'].mean()
            avg_weekday_stay = df['stays_in_week_nights'].mean()
            avg_adr = df['adr'].mean()
            
            # Top countries
            top_countries = df['country'].value_counts().head(3).to_dict()
            
            analytics = {
                "total_bookings": total_bookings,
                "cancellation_rate": round(canceled_rate, 2),
                "avg_lead_time_days": round(avg_lead_time, 2),
                "avg_total_stay_nights": round(avg_stay, 2),
                "avg_weekend_nights": round(avg_weekend_stay, 2),
                "avg_weekday_nights": round(avg_weekday_stay, 2),
                "avg_daily_rate": round(avg_adr, 2),
                "top_countries": top_countries
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error generating analytics: {str(e)}")
            return {"error": str(e)}

# Initialize FastAPI
app = FastAPI(title="Hotel Booking RAG API", 
              description="API for querying hotel booking data using Retrieval Augmented Generation")

# Initialize RAG system
CSV_PATH = os.getenv("CSV_PATH", "Data/processed_data1.csv")
rag_system = None

@app.on_event("startup")
async def startup_event():
    global rag_system
    try:
        rag_system = RAGSystem(CSV_PATH)
        logger.info("RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Endpoint to answer questions about hotel booking data"""
    global rag_system
    
    if rag_system is None:
        try:
            rag_system = RAGSystem(CSV_PATH)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"RAG system initialization failed: {str(e)}")
    
    try:
        result = rag_system.query(request.question, request.top_k)
        return {
            "question": request.question,
            "answer": result["answer"],
            "context": result.get("context", []),
            "performance": result.get("performance", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analytics")
async def get_analytics(request: AnalyticsRequest):
    """Endpoint to get analytics about hotel booking data"""
    global rag_system
    
    if rag_system is None:
        try:
            rag_system = RAGSystem(CSV_PATH)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"RAG system initialization failed: {str(e)}")
    
    try:
        start_time = time.time()
        analytics = rag_system.generate_analytics()
        processing_time = time.time() - start_time
        
        return {
            "report_type": request.report_type,
            "data": analytics,
            "performance": {
                "processing_time": round(processing_time, 3)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Endpoint to check API health"""
    return {
        "status": "healthy",
        "rag_system_initialized": rag_system is not None,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)