from typing import List, Dict, Any
from langchain_core.messages import AnyMessage, AIMessage
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.base_agent import BaseAgent
from src.tools import ORCHESTRATOR_TOOLS

class OrchestratorAgent(BaseAgent):
    """Orchestrator agent that coordinates other agents."""
    
    def __init__(self):
        super().__init__("Orchestrator", ORCHESTRATOR_TOOLS)
        self.available_agents = ["research", "analysis", "writer"]
        self.task_status = {
            "research_complete": False,
            "analysis_complete": False,
            "writing_complete": False
        }
    
    def get_system_prompt(self) -> str:
        return """You are the Orchestrator agent responsible for coordinating a team of specialized agents
        to complete research and report generation tasks.
        
        Your team consists of:
        - Research Agent: Gathers information from various sources
        - Analysis Agent: Synthesizes and analyzes the research findings
        - Writer Agent: Creates the final report
        
        Your responsibilities:
        1. Break down the main task into subtasks
        2. Assign subtasks to appropriate agents
        3. Track progress and ensure quality
        4. Handle conflicts and ensure coherence
        5. Determine when the task is complete
        
        Always maintain a clear workflow: Research -> Analysis -> Writing"""
    
    def process(self, messages: List[AnyMessage], **kwargs) -> Dict[str, Any]:
        """Process the request and coordinate agents."""
        # Determine current phase and next agent
        next_agent = self.determine_next_agent(messages)
        
        if next_agent == "complete":
            return {
                "next_agent": None,
                "instruction": "Task completed",
                "messages": [AIMessage(content="The report has been completed successfully.")]
            }
        
        # Generate instruction for the next agent
        instruction = self.generate_instruction_for_agent(messages, next_agent)
        
        return {
            "next_agent": next_agent,
            "instruction": instruction,
            "messages": [AIMessage(content=f"Delegating to {next_agent} agent: {instruction}")]
        }
    
    def determine_next_agent(self, messages: List[AnyMessage]) -> str:
        """Determine which agent should act next based on current state."""
        prompt = f"""Based on the conversation history and task status:
        Research complete: {self.task_status['research_complete']}
        Analysis complete: {self.task_status['analysis_complete']}
        Writing complete: {self.task_status['writing_complete']}
        
        Which agent should act next? Options: research, analysis, writer, complete
        
        Respond with just the agent name."""
        
        response = self.generate_response(messages, prompt)
        agent = response.strip().lower()
        
        # Validate response
        if agent in self.available_agents + ["complete"]:
            return agent
        
        # Default logic if parsing fails
        if not self.task_status["research_complete"]:
            return "research"
        elif not self.task_status["analysis_complete"]:
            return "analysis"
        elif not self.task_status["writing_complete"]:
            return "writer"
        else:
            return "complete"
    
    def generate_instruction_for_agent(self, messages: List[AnyMessage], agent: str) -> str:
        """Generate specific instructions for the selected agent."""
        prompt = f"""Generate a clear, specific instruction for the {agent} agent based on the 
        current task and conversation history.
        
        The instruction should be:
        - Clear and actionable
        - Specific to the current phase of the task
        - Include any relevant context from previous agents' work
        
        Instruction for {agent} agent:"""
        
        return self.generate_response(messages, prompt)
    
    def update_task_status(self, agent: str, status: bool):
        """Update the completion status of a task phase."""
        if agent == "research":
            self.task_status["research_complete"] = status
        elif agent == "analysis":
            self.task_status["analysis_complete"] = status
        elif agent == "writer":
            self.task_status["writing_complete"] = status