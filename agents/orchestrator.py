import os
from typing import Dict, Any, Optional
from agents.classifier import ClassifierAgent
from agents.json_agent import JSONAgent
from agents.email_agent import EmailAgent
from memory.manager import MemoryManager
from schema.models import ProcessingResponse, DocumentFormat

class AgentOrchestrator:
    """Orchestrates the multi-agent system using LangChain agents"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise ValueError("DEEPSEEK_API_KEY environment variable is required")

        self.classifier = ClassifierAgent(memory_manager)
        self.json_agent = JSONAgent(memory_manager)
        self.email_agent = EmailAgent(memory_manager)

        self.agent_map = {
            "json_agent": self.json_agent,
            "email_agent": self.email_agent,
        }
    
    async def process_document(
        self, 
        content: str, 
        content_type: str, 
        metadata: Optional[Dict[str, Any]] = None,
        model_id: Optional[str] = "deepseek-chat"
    ) -> ProcessingResponse:
        """Process a document through the multi-agent system"""
        
        try:
            classification_input = {
                "content": content,
                "content_type": content_type,
                "model_id": model_id
            }
            
            classification_result = await self.classifier.process(classification_input)
            
            if "error" in classification_result:
                return ProcessingResponse(
                    success=False,
                    message=f"Classification failed: {classification_result['error']}",
                    memory_id=classification_result.get("memory_id")
                )
            routing_target = classification_result["routing_target"]
            classification_data = classification_result["classification"]
            
            if routing_target not in self.agent_map:
                return ProcessingResponse(
                    success=False,
                    message=f"No agent available for routing target: {routing_target}",
                    memory_id=classification_result.get("memory_id")
                )
            
            target_agent = self.agent_map[routing_target]
            context = {
                "intent": classification_data["intent"],
                "format": classification_data["format"],
                "confidence": classification_data["confidence"],
                "classification_memory_id": classification_result["memory_id"]
            }
            
            if metadata:
                context.update(metadata)
            agent_input = {
                "content": content,
                "content_type": content_type,
                "model_id": model_id
            }
            
            processing_result = await target_agent.process(agent_input, context)
            
            if not processing_result.get("success", True):
                return ProcessingResponse(
                    success=False,
                    message=f"Agent processing failed: {processing_result.get('error', 'Unknown error')}",
                    memory_id=processing_result.get("memory_id")
                )
            return ProcessingResponse(
                success=True,
                message=f"Document successfully processed by {routing_target}",
                data={
                    "classification": classification_data,
                    "processing_result": processing_result.get("result"),
                    "routing_target": routing_target
                },
                memory_id=processing_result.get("memory_id")
            )
            
        except Exception as e:
            error_memory_id = await self.memory_manager.store_entry(
                source="orchestrator",
                document_type=DocumentFormat.EMAIL,
                extracted_values={
                    "error": "Orchestration error",
                    "error_details": str(e),
                    "content_preview": content[:500]
                }
            )
            
            return ProcessingResponse(
                success=False,
                message=f"System error: {str(e)}",
                memory_id=error_memory_id
            )
    
    async def get_processing_history(self, limit: int = 10) -> Dict[str, Any]:
        """Get recent processing history"""
        try:
            recent_entries = await self.memory_manager.get_recent_context(limit)
            
            history = []
            for entry in recent_entries:
                summary = "No summary available"
                
                if entry.source == "classifier_agent":
                    classification = entry.extracted_values.get("classification", {})
                    if classification:
                        summary = f"Classified as {classification.get('format', 'unknown')} format with {classification.get('intent', 'unknown')} intent (confidence: {classification.get('confidence', 0):.2f})"
                
                elif entry.source == "email_agent":
                    analysis = entry.extracted_values.get("analysis", {})
                    if analysis:
                        sender = analysis.get("sender", "unknown")
                        urgency = analysis.get("urgency", "unknown")
                        summary = f"Email from {sender} with {urgency} urgency: {analysis.get('crm_summary', 'No summary')}"
                
                elif entry.source == "json_agent":
                    analysis = entry.extracted_values.get("analysis", {})
                    if analysis:
                        validation = "valid" if analysis.get("validation_passed", False) else "invalid"
                        summary = f"JSON document ({validation}): {analysis.get('summary', 'No summary')}"
                
                # Fallback to error details if available
                if summary == "No summary available" and "error_details" in entry.extracted_values:
                    summary = f"Error: {entry.extracted_values['error_details'][:100]}..."
                
                history.append({
                    "id": entry.id,
                    "source": entry.source,
                    "document_type": entry.document_type.value,
                    "intent": entry.intent.value,
                    "timestamp": entry.timestamp.isoformat(),
                    "summary": summary
                })
            
            return {
                "success": True,
                "history": history,
                "total_entries": len(history)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to retrieve history: {str(e)}"
            }
    
    async def get_memory_entry(self, memory_id: str) -> Dict[str, Any]:
        """Get a specific memory entry"""
        try:
            entry = await self.memory_manager.get_entry(memory_id)
            if not entry:
                return {
                    "success": False,
                    "error": "Memory entry not found"
                }
            
            return {
                "success": True,
                "entry": entry.model_dump()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to retrieve memory entry: {str(e)}"
            }
