import json
from typing import Dict, Any, Optional
from langchain.prompts import ChatPromptTemplate
from agents.base import BaseAgent
from memory.manager import MemoryManager
from schema.models import DocumentFormat, DocumentIntent

class EmailAgent(BaseAgent):
    """Agent for processing email content"""
    
    def __init__(self, memory_manager: MemoryManager):
        super().__init__(memory_manager)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an email processing expert. Analyze the email content and provide detailed analysis including:

            1. Extract sender information and contact details
            2. Determine urgency level (CRITICAL/HIGH/MEDIUM/LOW)
            3. Analyze sentiment and intent
            4. Create CRM-ready summary
            5. Identify follow-up requirements

            CRITICAL: Your response MUST be ONLY a valid JSON object starting with {{ and ending with }}. Do NOT include any text before or after the JSON. Do NOT use markdown formatting.

            Respond with this exact JSON structure:
            {{
            "sender": "email@domain.com",
            "subject": "email subject",
            "urgency": "HIGH|MEDIUM|LOW|CRITICAL",
            "sentiment": "positive|negative|neutral",
            "key_points": ["point1", "point2"],
            "crm_summary": "Brief summary for CRM",
            "follow_up_required": true/false,
            "contact_info_extracted": {{}}
            }}

            Urgency indicators: urgent, asap, critical, emergency, down, broken
            High priority: important, priority, needed soon
            Medium: request, question, inquiry
            Low: general information, updates"""),
            ("user", "Intent: {intent}\n\nEmail Content:\n{content}"),
        ])
    
    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process email document"""
        content = input_data.get("content", "")
        model_id = input_data.get("model_id", "deepseek-chat")
        intent = context.get("intent", "general") if context else "general"
        
        try:
            messages = self.prompt.format_messages(
                intent=intent,
                content=content[:3000]
            )
            
            # Use the specified model
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
                    "sender": "unknown@example.com",
                    "subject": "Processing failed",
                    "urgency": "MEDIUM",
                    "sentiment": "neutral",
                    "key_points": [f"Failed to parse response: {json_err}"],
                    "crm_summary": "Email processing error",
                    "follow_up_required": False,
                    "contact_info_extracted": {}
                }

            memory_id = await self.memory_manager.store_entry(
                source="email_agent",
                document_type=DocumentFormat.EMAIL,
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
                source="email_agent",
                document_type=DocumentFormat.EMAIL,
                intent=DocumentIntent.GENERAL,
                extracted_values={
                    "error": "Email processing failed",
                    "error_details": str(e),
                    "content_preview": content[:500]
                }
            )
            
            return {
                "success": False,
                "error": f"Email processing failed: {str(e)}",
                "memory_id": memory_id
            }