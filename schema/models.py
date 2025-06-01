from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class ModelProvider(str, Enum):
    """Enum for supported model providers"""
    
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"


class SupportedModel(str, Enum):
    """Supported model IDs"""
    
    GPT_35_TURBO = "gpt-3.5-turbo"
    O1_PREVIEW = "o1-preview"
    CLAUDE_35_HAIKU_LATEST = "claude-3-5-haiku-latest"
    CLAUDE_35_SONNET_LATEST = "claude-3-5-sonnet-latest"
    GPT_4O_MINI = "gpt-4o-mini"
    O1 = "o1"
    GPT_4O = "gpt-4o"
    CLAUDE_37_SONNET_LATEST = "claude-3-7-sonnet-latest"
    DEEPSEEK_CHAT = "deepseek-chat"
    DEEPSEEK_REASON = "deepseek-reason"
    
    @classmethod
    def get_provider(cls, model_id: str) -> ModelProvider:
        """Get the provider for a model ID"""
        if model_id.startswith(("gpt-", "o1")):
            return ModelProvider.OPENAI
        elif model_id.startswith("claude-"):
            return ModelProvider.ANTHROPIC
        elif model_id.startswith("deepseek-"):
            return ModelProvider.DEEPSEEK
        else:
            raise ValueError(f"Unsupported model ID: {model_id}")

class DocumentFormat(str, Enum):
    PDF = "pdf"
    JSON = "json"
    EMAIL = "email"

class DocumentIntent(str, Enum):
    INVOICE = "invoice"
    RFQ = "rfq"
    COMPLAINT = "complaint"
    REGULATION = "regulation"
    GENERAL = "general"

class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ClassificationResult(BaseModel):
    format: DocumentFormat
    intent: DocumentIntent
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)

class MemoryEntry(BaseModel):
    id: str
    source: str
    document_type: DocumentFormat
    intent: DocumentIntent
    timestamp: datetime = Field(default_factory=datetime.now)
    extracted_values: Dict[str, Any] = Field(default_factory=dict)
    thread_id: Optional[str] = None
    conversation_id: Optional[str] = None

class EmailData(BaseModel):
    sender: str
    subject: str
    body: str
    recipients: List[str] = Field(default_factory=list)
    timestamp: Optional[datetime] = None

class EmailProcessingResult(BaseModel):
    sender: str
    intent: DocumentIntent
    urgency: UrgencyLevel
    extracted_info: Dict[str, Any]
    formatted_for_crm: Dict[str, Any]

class JSONProcessingResult(BaseModel):
    validated_data: Dict[str, Any]
    anomalies: List[str] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)
    reformatted_data: Dict[str, Any]

class ProcessingRequest(BaseModel):
    content: str
    content_type: str
    metadata: Optional[Dict[str, Any]] = None
    model_id: Optional[SupportedModel] = Field(default=SupportedModel.DEEPSEEK_CHAT, description="Model to use for processing")

class ProcessingResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    memory_id: Optional[str] = None
