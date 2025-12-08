"""
Structured Output Agent Implementation

Agent with structured output support using LangChain's structured output feature.
Supports both langchain>=1.1 (create_agent with response_format) and 
langchain<1.1 (with_structured_output method).

Based on: https://docs.langchain.com/oss/python/langchain/structured-output
"""

from typing import Any, Dict, Optional, Type, Union
from pydantic import BaseModel

try:
    from agentensemble.utils.llm import get_mistral_model
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    get_mistral_model = None

# Try langchain>=1.1 approach (create_agent with response_format)
try:
    from langchain.agents import create_agent
    from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
    CREATE_AGENT_AVAILABLE = True
    STRUCTURED_OUTPUT_AVAILABLE = True
except ImportError:
    CREATE_AGENT_AVAILABLE = False
    create_agent = None
    ToolStrategy = None
    ProviderStrategy = None
    STRUCTURED_OUTPUT_AVAILABLE = False

# with_structured_output is available if model has the method
WITH_STRUCTURED_AVAILABLE = True


class StructuredAgent:
    """
    Agent with structured output support.
    
    Uses LangChain's structured output feature to return data in a specific format.
    
    For langchain>=1.1: Uses create_agent with response_format parameter
    For langchain<1.1: Uses model.with_structured_output() method
    
    Based on: https://docs.langchain.com/oss/python/langchain/structured-output
    """
    
    def __init__(
        self,
        name: str = "structured_agent",
        tools: Optional[list] = None,
        model: Optional[Any] = None,
        response_format: Optional[Union[Type[BaseModel], Any]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize structured agent.
        
        Args:
            name: Agent name
            tools: Available tools
            model: LLM model (defaults to Mistral if not provided)
            response_format: Structured output schema (Pydantic model)
            system_prompt: System prompt for the agent
            **kwargs: Additional configuration
        """
        if not LLM_AVAILABLE:
            raise ImportError(
                "langchain-mistralai is required for StructuredAgent. "
                "Install with: pip install langchain-mistralai"
            )
        
        self.name = name
        self.tools = tools or []
        self.model = model or get_mistral_model()
        self.response_format = response_format
        self.system_prompt = system_prompt
        self.config = kwargs
        
        # Determine which approach to use
        if CREATE_AGENT_AVAILABLE and response_format:
            # Use create_agent with response_format (langchain>=1.1)
            self._use_create_agent()
        elif response_format and hasattr(self.model, 'with_structured_output'):
            # Use with_structured_output (langchain<1.1)
            self._use_with_structured_output()
        else:
            # Basic agent without structured output
            if CREATE_AGENT_AVAILABLE:
                self._use_create_agent()
            else:
                raise ImportError(
                    "langchain>=1.1 is required for StructuredAgent with tools. "
                    "Install with: pip install 'langchain>=1.1'"
                )
    
    def _use_create_agent(self):
        """Use create_agent approach (langchain>=1.1)"""
        if not CREATE_AGENT_AVAILABLE:
            raise ImportError("create_agent not available. Requires langchain>=1.1")
        
        agent_kwargs = {
            "model": self.model,
            "tools": self.tools,
        }
        
        if self.system_prompt:
            agent_kwargs["system_prompt"] = self.system_prompt
        
        if self.response_format:
            agent_kwargs["response_format"] = self.response_format
        
        agent_kwargs.update(self.config)
        self.agent = create_agent(**agent_kwargs)
        self._method = "create_agent"
    
    def _use_with_structured_output(self):
        """Use with_structured_output approach (langchain<1.1)"""
        if not self.response_format:
            raise ValueError("response_format required for with_structured_output")
        
        # Bind structured output to model
        self.structured_model = self.model.with_structured_output(self.response_format)
        self._method = "with_structured_output"
    
    def run(
        self,
        query: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute agent with structured output.
        
        Args:
            query: Input query
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing:
            - 'result': Natural language response
            - 'structured_response': Structured output (if response_format provided)
            - 'metadata': Execution metadata
        """
        if self._method == "create_agent":
            # Use create_agent approach
            result = self.agent.invoke({
                "messages": [{"role": "user", "content": query}],
                **kwargs
            })
            
            # Extract structured response if available
            structured_response = result.get("structured_response")
            
            # Get last message content as natural language result
            messages = result.get("messages", [])
            last_message = messages[-1] if messages else None
            natural_result = ""
            
            if last_message:
                if hasattr(last_message, "content"):
                    natural_result = last_message.content
                elif isinstance(last_message, dict):
                    natural_result = last_message.get("content", "")
            
            return {
                "result": natural_result or "No result generated",
                "structured_response": structured_response,
                "metadata": {
                    "agent": self.name,
                    "has_structured_output": structured_response is not None,
                    "method": "create_agent"
                }
            }
        
        elif self._method == "with_structured_output":
            # Use with_structured_output approach
            structured_response = self.structured_model.invoke(query)
            
            return {
                "result": str(structured_response),
                "structured_response": structured_response,
                "metadata": {
                    "agent": self.name,
                    "has_structured_output": True,
                    "method": "with_structured_output"
                }
            }
        
        else:
            raise ValueError(f"Unknown method: {self._method}")
    
    def invoke(self, input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Direct invoke method for compatibility with LangChain agents.
        
        Args:
            input: Input dictionary with messages
            **kwargs: Additional parameters
            
        Returns:
            Agent result with structured_response if available
        """
        if self._method == "create_agent":
            return self.agent.invoke(input, **kwargs)
        else:
            query = input.get("messages", [{}])[-1].get("content", "") if isinstance(input.get("messages"), list) else str(input)
            return self.run(query, **kwargs)
