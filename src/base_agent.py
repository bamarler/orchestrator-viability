from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
import google.generativeai as genai
from dotenv import load_dotenv
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logging_utils import AgentLogger, log_tool_execution

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=api_key)

class BaseAgent(ABC):
    """Abstract base class for all agents in the multi-agent system."""
    
    def __init__(self, name: str, tools: List, model_name: str = 'gemini-2.0-flash-lite'):
        self.name = name
        self.tools = tools
        self.model = genai.GenerativeModel(model_name)
        self.memory = []  # Agent's local memory
        self.logger = None  # Will be set by the graph
        
    @abstractmethod
    def process(self, messages: List[AnyMessage], **kwargs) -> Dict[str, Any]:
        """
        Process incoming messages and return a response.
        
        Args:
            messages: List of messages from conversation history
            **kwargs: Additional parameters specific to agent type
            
        Returns:
            Dict containing response and any additional data
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        pass
    
    def set_logger(self, logger: AgentLogger):
        """Set the logger for this agent."""
        self.logger = logger
    
    def log(self, message: str, metadata: Optional[dict] = None):
        """Log a message if logger is available."""
        if self.logger:
            self.logger.log(self.name, message, metadata)
    
    def generate_response(self, messages: List[AnyMessage], additional_prompt: str = "") -> str:
        """Generate a response using the LLM."""
        time.sleep(1)
        system_prompt = self.get_system_prompt()
        
        # Combine system prompt with message history
        full_prompt = f"{system_prompt}\n\n"
        
        if additional_prompt:
            full_prompt += f"{additional_prompt}\n\n"
            
        # Add conversation history
        for msg in messages:
            if isinstance(msg, HumanMessage):
                full_prompt += f"Human: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                full_prompt += f"Assistant: {msg.content}\n"
        
        self.log("Generating response", {"prompt_length": len(full_prompt)})
        
        try:
            response = self.model.generate_content(full_prompt)
            result = response.text.strip()
            self.log("Response generated", {"response_length": len(result)})
            return result
        except Exception as e:
            self.log(f"Error generating response: {str(e)}", {"error": str(e)})
            return f"Error generating response: {str(e)}"
    
    def execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Execute a tool by name with given input."""
        tool_map = {tool.name: tool for tool in self.tools}
        
        if tool_name not in tool_map:
            error_msg = f"Tool '{tool_name}' not available for {self.name}"
            self.log(error_msg, {"available_tools": [t.name for t in self.tools]})
            return error_msg
            
        try:
            result = tool_map[tool_name].invoke(tool_input)
            if self.logger:
                log_tool_execution(self.logger, self.name, tool_name, tool_input, result)
            return result
        except Exception as e:
            error_msg = f"Error executing {tool_name}: {str(e)}"
            self.log(error_msg, {"error": str(e)})
            return error_msg
    
    def plan_next_action(self, messages: List[AnyMessage], context: Optional[str] = None) -> Dict[str, str]:
        """Plan the next action based on current state."""
        # Get tool descriptions
        tool_descriptions = []
        for tool in self.tools:
            tool_descriptions.append(f"- {tool.name}: {tool.description if hasattr(tool, 'description') else 'Available tool'}")
        
        planning_prompt = f"""Based on the conversation and your role as {self.name}, 
        decide what action to take next.
        
        Current context: {context or 'Processing task'}
        
        Available tools: {chr(10).join(tool_descriptions)}
        
        You can also choose 'respond' to provide a direct response without using tools.
        
        Consider:
        1. What information or action is needed next?
        2. Which tool would be most appropriate?
        3. What specific input should be provided to the tool?
        
        Respond in EXACTLY this format:
        ACTION: [exact tool name or 'respond']
        INPUT: [tool input or response text]
        REASONING: [brief explanation of why this action]"""
        
        response = self.generate_response(messages, planning_prompt)
        
        # Parse response with better error handling
        lines = response.split('\n')
        result = {'action': 'respond', 'input': '', 'reasoning': ''}
        
        for line in lines:
            line = line.strip()
            if line.startswith('ACTION:'):
                result['action'] = line.replace('ACTION:', '').strip()
            elif line.startswith('INPUT:'):
                result['input'] = line.replace('INPUT:', '').strip()
            elif line.startswith('REASONING:'):
                result['reasoning'] = line.replace('REASONING:', '').strip()
        
        self.log(f"Planned action: {result['action']}", result)
        return result
    
    def update_memory(self, key: str, value: Any):
        """Update agent's local memory."""
        self.memory.append({key: value})
    
    def get_memory(self) -> List[Dict]:
        """Get agent's local memory."""
        return self.memory
    
    def __repr__(self):
        return f"{self.name}(tools={[t.name for t in self.tools]})"