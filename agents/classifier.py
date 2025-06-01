import json
from typing import Dict, Any, Optional
from langchain.prompts import ChatPromptTemplate
from agents.base import BaseAgent
from memory.manager import MemoryManager
from schema.models import DocumentFormat, DocumentIntent

class ClassifierAgent(BaseAgent):
    """LangChain agent for document classification"""
    
    def __init__(self, memory_manager: MemoryManager):
        super().__init__(memory_manager)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a document classification expert. Analyze the raw content and determine both format and intent. 

            CRITICAL: Your response MUST be ONLY a valid JSON object starting with {{ and ending with }}. Do NOT include any text before or after the JSON. Do NOT use markdown formatting. Do NOT add explanatory text.

            Respond with this exact JSON structure:

            {{
            "format": "pdf" | "json" | "email",
            "intent": "invoice" | "rfq" | "complaint" | "regulation" | "general", 
            "confidence": 0.0-1.0,
            "reasoning": "explanation of your decision",
            "routing_target": "json_agent" | "email_agent"
            }}

            Format Classification (analyze the content structure):
            - json: Valid JSON structure with {{}}, [], proper syntax
            - email: Has email headers (From:, To:, Subject:) or email formatting
            - pdf: Text extracted from PDF or mentions PDF format

            Intent Classification (analyze the meaning):
            - invoice: Contains billing/payment terms, amounts, vendor info, invoice numbers
            - rfq: Request for quote, proposals, bidding requirements, procurement
            - complaint: Issues, problems, urgent matters, service complaints, system down
            - regulation: Policies, compliance, regulatory requirements, legal documents
            - general: Default for other content

            Routing Rules:
            - Route json format → json_agent
            - Route email/pdf format → email_agent

            REMEMBER: Return ONLY the JSON object, nothing else!"""),
            ("user", "Analyze this content and classify it:\n\n{content}"),
        ])
    
    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Classify the input document using direct LLM calls"""
        content = input_data.get("content", "")
        
        try:
            messages = self.prompt.format_messages(
                content=content[:2000] 
            )
            
            response = await self.llm.ainvoke(messages)

            try:
                raw_content = response.content.strip()

                if "```json" in raw_content:
                    start = raw_content.find("```json") + 7
                    end = raw_content.find("```", start)
                    raw_content = raw_content[start:end].strip()
                elif raw_content.startswith("```"):
                    lines = raw_content.split('\n')
                    raw_content = '\n'.join(lines[1:-1])

                if not raw_content.startswith('{'):
                    start_idx = raw_content.find('{')
                    if start_idx != -1:
                        raw_content = raw_content[start_idx:]
                
                if not raw_content.endswith('}'):
                    end_idx = raw_content.rfind('}')
                    if end_idx != -1:
                        raw_content = raw_content[:end_idx + 1]
                
                classification_data = json.loads(raw_content)
                
            except json.JSONDecodeError as json_err:
                classification_data = {
                    "format": "EMAIL",
                    "intent": "GENERAL",
                    "confidence": 0.5,
                    "reasoning": f"Failed to parse LLM response as JSON: {json_err}",
                    "routing_target": "email_agent"
                }

            memory_id = await self.memory_manager.store_entry(
                source="classifier_agent",
                document_type=DocumentFormat(classification_data["format"]),
                intent=DocumentIntent(classification_data["intent"]),
                extracted_values={
                    "classification": classification_data,
                    "content_preview": content[:500]
                }
            )
            
            return {
                "success": True,
                "classification": classification_data,
                "routing_target": classification_data["routing_target"],
                "memory_id": memory_id
            }
                
        except Exception as e:
            memory_id = await self.memory_manager.store_entry(
                source="classifier_agent",
                document_type=DocumentFormat.EMAIL,
                intent=DocumentIntent.GENERAL,
                extracted_values={
                    "error": "Classification failed",
                    "error_details": str(e),
                    "content_preview": content[:500]
                }
            )
            
            return {
                "success": False,
                "error": f"Classification failed: {str(e)}",
                "routing_target": "email_agent", 
                "memory_id": memory_id
            }
