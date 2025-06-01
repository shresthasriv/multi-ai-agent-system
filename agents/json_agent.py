import json
from typing import Dict, Any, Optional
from langchain.prompts import ChatPromptTemplate
from agents.base import BaseAgent
from memory.manager import MemoryManager
from schema.models import DocumentFormat, DocumentIntent

class JSONAgent(BaseAgent):
    """Agent for processing JSON documents"""
    
    def __init__(self, memory_manager: MemoryManager):
        super().__init__(memory_manager)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a JSON document processing expert. Analyze the JSON content and provide a detailed analysis including:

            1. Validation status (is it valid JSON?)
            2. Missing required fields (based on document intent)
            3. Data anomalies (null values, negative amounts, etc.)
            4. Reformatted/cleaned version
            5. Summary and insights

            CRITICAL: Your response MUST be ONLY a valid JSON object starting with {{ and ending with }}. Do NOT include any text before or after the JSON. Do NOT use markdown formatting.

            Respond with this exact JSON structure:
            {{
            "validation_passed": true/false,
            "missing_fields": ["field1", "field2"],
            "anomalies": ["description of issues found"],
            "reformatted_data": {{}},
            "summary": "Brief summary of the document",
            "key_insights": ["insight1", "insight2"]
            }}

            For invoices: check for amount, vendor, date, items
            For RFQs: check for requirements, deadline, contact info
            For complaints: check for issue description, severity
            For regulations: check for compliance requirements"""),
            ("user", "Intent: {intent}\n\nJSON Content:\n{content}"),
        ])
    
    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process JSON document"""
        content = input_data.get("content", "")
        model_id = input_data.get("model_id", "deepseek-chat")
        intent = context.get("intent", "general") if context else "general"
        
        try:
            messages = self.prompt.format_messages(
                intent=intent,
                content=content[:3000]
            )

            llm = self.get_llm(model_id)
            response = await llm.ainvoke(messages)

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
                
                analysis_result = json.loads(raw_content)
                
            except json.JSONDecodeError as json_err:
                analysis_result = {
                    "validation_passed": False,
                    "missing_fields": [],
                    "anomalies": [f"Failed to parse LLM response: {json_err}"],
                    "reformatted_data": {},
                    "summary": "Processing failed",
                    "key_insights": []
                }

            memory_id = await self.memory_manager.store_entry(
                source="json_agent",
                document_type=DocumentFormat.JSON,
                intent=DocumentIntent(intent.lower()),
                extracted_values={
                    "analysis": analysis_result,
                    "original_content": content[:1000]
                }
            )
            
            return {
                "success": True,
                "result": analysis_result,
                "memory_id": memory_id
            }
                
        except Exception as e:
            memory_id = await self.memory_manager.store_entry(
                source="json_agent",
                document_type=DocumentFormat.JSON,
                intent=DocumentIntent.GENERAL,
                extracted_values={
                    "error": "JSON processing failed",
                    "error_details": str(e),
                    "content_preview": content[:500]
                }
            )
            
            return {
                "success": False,
                "error": f"JSON processing failed: {str(e)}",
                "memory_id": memory_id
            }
