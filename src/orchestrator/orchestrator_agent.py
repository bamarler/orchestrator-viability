from typing import List, Dict, Any, Optional
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.base_agent import BaseAgent
from src.tools import ORCHESTRATOR_TOOLS

class OrchestratorAgent(BaseAgent):
    """Orchestrator agent that coordinates other agents using dynamic tool selection."""
    
    def __init__(self):
        super().__init__("Orchestrator", ORCHESTRATOR_TOOLS)
        self.available_agents = ["research", "analysis", "writer"]
        self.task_status = {
            "research_complete": False,
            "analysis_complete": False, 
            "writing_complete": False
        }
        self.max_delegation_iterations = 10
        self.current_iteration = 0
    
    def get_system_prompt(self) -> str:
        return """You are the Orchestrator agent responsible for coordinating a team of specialized agents
        to complete research and report generation tasks.
        
        Your team consists of:
        - Research Agent: Gathers comprehensive information from various sources
        - Analysis Agent: Synthesizes and analyzes the research findings
        - Writer Agent: Creates the final report with proper structure
        
        Your responsibilities:
        1. Break down the main task into clear subtasks
        2. Assign subtasks to appropriate agents with specific instructions
        3. Track progress using memory tools
        4. Ensure quality by reviewing agent outputs
        5. Request more work if outputs are insufficient
        6. Determine when the task is complete
        
        Use your tools to:
        - Check memory for agent outputs
        - Track task progress
        - Make informed delegation decisions
        
        Be demanding - ensure each phase produces thorough, high-quality output.
        Maintain workflow: Research -> Analysis -> Writing, but loop back if needed."""
    
    def process(self, messages: List[AnyMessage], **kwargs) -> Dict[str, Any]:
        """Process the request and coordinate agents using dynamic planning."""
        self.current_iteration += 1
        
        # Check overall progress
        progress_summary = self.check_progress()
        
        # Plan next action using inherited method
        context = f"Progress: {progress_summary}, Iteration: {self.current_iteration}/{self.max_delegation_iterations}"
        action_plan = self.plan_next_action(messages, context)
        
        # If agent decides to use a tool
        if action_plan['action'] != 'respond':
            tool_result = self.execute_tool(action_plan['action'], action_plan['input'])
            messages.append(AIMessage(content=f"Tool result: {tool_result}"))
            # Recursively call to make next decision
            return self.process(messages, **kwargs)
        
        # Otherwise, make delegation decision
        next_agent = self.determine_next_agent_dynamically(messages, progress_summary)
        
        if next_agent == "complete":
            # Verify completion
            final_check = self.verify_completion()
            if final_check['is_complete']:
                return {
                    "next_agent": None,
                    "instruction": "Task completed successfully",
                    "messages": [AIMessage(content=f"Task completed! {final_check['summary']}")]
                }
            else:
                # Force completion if max iterations reached
                if self.current_iteration >= self.max_delegation_iterations:
                    return {
                        "next_agent": None,
                        "instruction": "Task completed (max iterations reached)",
                        "messages": [AIMessage(content="Task completed after maximum iterations.")]
                    }
                # Otherwise, identify what's missing
                next_agent = final_check['needed_agent']
        
        # Generate specific, detailed instruction for the next agent
        instruction = self.generate_detailed_instruction(messages, next_agent, progress_summary)
        
        return {
            "next_agent": next_agent,
            "instruction": instruction,
            "messages": [AIMessage(content=f"Delegating to {next_agent} agent with detailed instruction: {instruction[:200]}...")]
        }
    
    def check_progress(self) -> str:
        """Check progress by examining memory."""
        progress_parts = []
        
        # Use tools to check what's in memory
        memory_keys = self.execute_tool('list_memory_keys', '')
        
        # Check for key outputs
        if 'research_synthesis' in memory_keys or 'research_summary' in memory_keys:
            self.task_status['research_complete'] = True
            progress_parts.append("Research: DONE")
        else:
            progress_parts.append("Research: PENDING")
            
        if 'analysis_insights' in memory_keys or 'analysis_complete' in memory_keys:
            self.task_status['analysis_complete'] = True
            progress_parts.append("Analysis: DONE")
        else:
            progress_parts.append("Analysis: PENDING")
            
        if 'final_report' in memory_keys:
            self.task_status['writing_complete'] = True
            progress_parts.append("Writing: DONE")
        else:
            progress_parts.append("Writing: PENDING")
        
        return ", ".join(progress_parts)
    
    def determine_next_agent_dynamically(self, messages: List[AnyMessage], progress: str) -> str:
        """Dynamically determine next agent based on current state."""
        decision_prompt = f"""Based on current progress: {progress}
        
        And the task requirements from the conversation, which agent should act next?
        
        Consider:
        1. Has research provided comprehensive information?
        2. Has analysis produced actionable insights?
        3. Has writing created a complete report meeting requirements?
        4. Should we loop back to improve any phase?
        
        Options: research, analysis, writer, complete
        
        Choose the most appropriate next step. Be critical - if any phase seems weak, loop back.
        
        Respond with just the agent name or 'complete'."""
        
        response = self.generate_response(messages, decision_prompt).strip().lower()
        
        # Validate response
        if response in self.available_agents + ["complete"]:
            return response
        
        # Fallback logic based on status
        if not self.task_status["research_complete"]:
            return "research"
        elif not self.task_status["analysis_complete"]:
            return "analysis"
        elif not self.task_status["writing_complete"]:
            return "writer"
        else:
            return "complete"
    
    def generate_detailed_instruction(self, messages: List[AnyMessage], agent: str, progress: str) -> str:
        """Generate detailed, specific instructions for the selected agent."""
        # Extract original task
        original_task = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                original_task = msg.content
                break
        
        instruction_prompt = f"""Generate detailed instructions for the {agent} agent.
        
        Original task: {original_task}
        Current progress: {progress}
        
        The instruction should:
        1. Be specific about what needs to be done
        2. Reference any previous work that should be built upon
        3. Set clear quality expectations
        4. Specify the expected output format
        5. Mention any particular areas to focus on
        
        For {agent} agent, the instruction should emphasize:"""
        
        if agent == "research":
            instruction_prompt += """
        - Creating a comprehensive outline first
        - Finding diverse, credible sources
        - Covering all aspects of the topic
        - Saving organized findings to memory
        - Ensuring sufficient depth for a quality report"""
        elif agent == "analysis":
            instruction_prompt += """
        - Reading all research from memory
        - Identifying key patterns and insights
        - Checking data consistency
        - Providing actionable conclusions
        - Highlighting any gaps that need addressing"""
        elif agent == "writer":
            instruction_prompt += """
        - Creating COMPLETE content, not outlines
        - Following proper report structure
        - Meeting word count requirements
        - Using all available research and analysis
        - Ensuring professional quality"""
        
        instruction_prompt += "\n\nDetailed instruction:"
        
        return self.generate_response(messages, instruction_prompt)
    
    def verify_completion(self) -> Dict[str, Any]:
        """Verify if the task is truly complete with high standards."""
        # Check final report quality
        report = self.execute_tool('read_from_memory', 'final_report')
        
        if "No data found" in report:
            return {
                'is_complete': False,
                'needed_agent': 'writer',
                'reason': 'No final report found'
            }
        
        verification_prompt = f"""Assess if this task is complete to high standards:
        
        Report preview: {report[:500]}...
        
        Check:
        1. Is there a complete report (not just an outline)?
        2. Does it meet professional quality standards?
        3. Are all sections properly developed?
        4. Is the research comprehensive?
        5. Are insights actionable?
        
        If NOT complete, which agent should improve it? (research/analysis/writer)
        
        Format response:
        COMPLETE: [YES/NO]
        NEEDED_AGENT: [agent name or NONE]
        REASON: [brief explanation]"""
        
        response = self.generate_response([], verification_prompt)
        
        is_complete = "COMPLETE: YES" in response
        needed_agent = "writer"  # default
        
        for line in response.split('\n'):
            if line.startswith('NEEDED_AGENT:'):
                agent = line.replace('NEEDED_AGENT:', '').strip().lower()
                if agent in self.available_agents:
                    needed_agent = agent
        
        return {
            'is_complete': is_complete,
            'needed_agent': needed_agent,
            'summary': 'All phases completed successfully.' if is_complete else f'Need {needed_agent} to improve output.'
        }
    
    def update_task_status(self, agent: str, status: bool):
        """Update the completion status of a task phase."""
        if agent == "research":
            self.task_status["research_complete"] = status
        elif agent == "analysis":
            self.task_status["analysis_complete"] = status
        elif agent == "writer":
            self.task_status["writing_complete"] = status