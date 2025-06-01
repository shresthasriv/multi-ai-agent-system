import os
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import aiofiles
import json
from agents.orchestrator import AgentOrchestrator
from memory.manager import MemoryManager
from schema.models import ProcessingResponse, SupportedModel, ModelProvider
from dotenv import load_dotenv
load_dotenv()



orchestrator: Optional[AgentOrchestrator] = None
memory_manager: Optional[MemoryManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    global orchestrator, memory_manager

    print("🚀 Starting Multi-Agent AI System...")

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    memory_manager = MemoryManager(redis_url)
    await memory_manager.connect()

    if not os.getenv("DEEPSEEK_API_KEY"):
        print("⚠️  DEEPSEEK_API_KEY not found. Please set it as an environment variable.")
        raise ValueError("DEEPSEEK_API_KEY is required for the system to function")

    orchestrator = AgentOrchestrator(memory_manager)
    print("✅ Multi-Agent System with LangChain and DeepSeek initialized successfully")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Multi-Agent AI System...")
    if memory_manager:
        await memory_manager.disconnect()
    print("✅ Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Multi-Agent AI Document Processing System",
    description="A system that classifies and processes documents using multiple specialized AI agents",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextProcessingRequest(BaseModel):
    content: str
    metadata: Optional[dict] = None
    model_id: Optional[str] = "deepseek-chat"


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Multi-Agent AI Document Processing System",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    global orchestrator, memory_manager
    
    status = {
        "system": "healthy",
        "components": {
            "orchestrator": orchestrator is not None,
            "memory_manager": memory_manager is not None,
        }
    }
    
    if memory_manager:
        try:
            # Test Redis connection
            await memory_manager.redis_client.ping()
            status["components"]["redis"] = True
        except Exception:
            status["components"]["redis"] = False
            status["system"] = "degraded"
    
    return status


@app.post("/process/text", response_model=ProcessingResponse)
async def process_text(request: TextProcessingRequest):
    """Process text content (email, JSON as string, etc.)"""
    global orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        result = await orchestrator.process_document(
            content=request.content,
            content_type="auto",
            metadata=request.metadata,
            model_id=request.model_id
        )
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/process/file", response_model=ProcessingResponse)
async def process_file(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    model_id: Optional[str] = Form("deepseek-chat")
):
    """Process uploaded file (PDF, JSON file, email file)"""
    global orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        content = await file.read()
        content_text = content.decode('utf-8')
        file_metadata = {}
        if metadata:
            try:
                file_metadata = json.loads(metadata)
            except Exception:
                file_metadata = {"raw_metadata": metadata}
        
        file_metadata["filename"] = file.filename
        file_metadata["content_type"] = file.content_type
        
        result = await orchestrator.process_document(
            content=content_text,
            content_type=file.content_type or "application/octet-stream",
            metadata=file_metadata,
            model_id=model_id
        )
        
        return result
    
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File content is not text-readable")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")


@app.get("/history")
async def get_processing_history(limit: int = 10):
    """Get recent processing history"""
    global orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        history = await orchestrator.get_processing_history(limit)
        return history
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")


@app.get("/memory/{memory_id}")
async def get_memory_entry(memory_id: str):
    """Get a specific memory entry"""
    global orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        entry = await orchestrator.get_memory_entry(memory_id)
        if not entry["success"]:
            raise HTTPException(status_code=404, detail=entry["error"])
        
        return entry
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve memory entry: {str(e)}")


@app.post("/classify")
async def classify_content(request: TextProcessingRequest):
    """Classify content without full processing"""
    global orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        classification_input = {
            "content": request.content,
            "model_id": request.model_id
        }
        
        result = await orchestrator.classifier.process(classification_input)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Multi-Agent AI Document Processing System...")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )