import os
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import uvicorn
from dotenv import load_dotenv

# Import RAG system from the separate file
from RAG import RAGSystem

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define API request/response models
class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class AnalyticsRequest(BaseModel):
    report_type: Optional[str] = "summary"

# Initialize FastAPI
app = FastAPI(title="Hotel Booking RAG API", 
              description="API for querying hotel booking data using Retrieval Augmented Generation")

# Initialize RAG system
CSV_PATH = os.getenv("CSV_PATH", "Data/processed_data.csv")
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
        import time
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
    uvicorn.run("FastAPI_app:app", host="0.0.0.0", port=port, reload=False)