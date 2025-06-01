import os
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor
from memory.manager import MemoryManager
import langchain
langchain.debug = False
langchain.verbose = False

# Set up logging for API calls
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.INFO)

# Also set up detailed logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all LangChain agents in the system"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        
        # Initialize default ChatDeepSeek with API key from environment
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if not deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is required")
            
        self.default_llm = ChatDeepSeek(
            api_key=deepseek_api_key,
            model="deepseek-chat",
            temperature=0.1
        )
        
        # Will be initialized by subclasses
        self.agent_executor: Optional[AgentExecutor] = None
    
    def get_llm(self, model_id: str = "deepseek-chat"):
        """Get LLM instance based on model_id"""
        if model_id.startswith("deepseek-"):
            return ChatDeepSeek(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                model=model_id,
                temperature=0.1
            )
        elif model_id.startswith(("gpt-", "o1")):
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                logger.warning("OPENAI_API_KEY not found, falling back to DeepSeek")
                return self.default_llm
            return ChatOpenAI(
                api_key=openai_api_key,
                model=model_id,
                temperature=0.1
            )
        elif model_id.startswith("claude-"):
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            if not anthropic_api_key:
                logger.warning("ANTHROPIC_API_KEY not found, falling back to DeepSeek")
                return self.default_llm
            return ChatAnthropic(
                api_key=anthropic_api_key,
                model=model_id,
                temperature=0.1
            )
        else:
            logger.warning(f"Model {model_id} not supported, falling back to deepseek-chat")
            return self.default_llm
    
    @property 
    def llm(self):
        """Backward compatibility property"""
        return self.default_llm
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process input data using the LangChain agent"""
        pass
