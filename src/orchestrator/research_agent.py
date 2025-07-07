from typing import List, Dict, Any
from langchain_core.messages import AnyMessage, AIMessage
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.base_agent import BaseAgent
from src.tools import RESEARCH_TOOLS

class ResearchAgent(BaseAgent):
    """Research agent for the orchestrator team."""
    
    def __init__(self):
        super().__init__("Research Agent (Orchestrator Team)", RESEARCH_TOOLS)
        self.research_complete = False
        self.sources_found = []
    
    def get_system_prompt(self) -> str:
        return """You are a Research Agent working under an Orchestrator's guidance.
        
        Your responsibilities:
        1. Search for relevant and credible information based on the given topic
        2. Extract key facts and data points
        3. Verify information across multiple sources
        4. Save important findings to shared memory for other agents
        5. Track sources for citation purposes
        
        Always aim for comprehensive coverage while maintaining accuracy.
        Report back to the Orchestrator when research is complete."""
    
    def process(self, messages: List[AnyMessage], instruction: str = None, **kwargs) -> Dict[str, Any]:
        """Execute research based on orchestrator's instruction."""
        # Plan research strategy
        research_plan = self.plan_research(messages, instruction)
        
        # Execute research
        findings = []
        for step in research_plan:
            if step['action'] == 'search':
                result = self.execute_tool('web_search', step['query'])
                findings.append({
                    'query': step['query'],
                    'result': result
                })
                
                # Extract and save key facts
                facts = self.execute_tool('extract_facts', result)

                self.execute_tool('save_to_memory', f"research_{step['query']}::{facts}")
                
            elif step['action'] == 'save':
                self.execute_tool('save_to_memory', f"{step['key']}::{step['value']}")
        
        # Generate research summary
        summary = self.generate_research_summary(findings)
        
        # Save summary to memory
        self.execute_tool('save_to_memory', f"research_summary::{summary}")
        
        return {
            "complete": True,
            "messages": [AIMessage(content=f"Research completed. Summary:\n{summary}")],
            "findings": findings
        }
    
    def plan_research(self, messages: List[AnyMessage], instruction: str) -> List[Dict]:
        """Plan research steps based on instruction."""
        prompt = f"""Given the instruction: {instruction}
        
        Plan 3-5 research steps. Each step should specify:
        - action: 'search' or 'save'
        - query: (for search) what to search for
        - key/value: (for save) what to save to memory
        
        Format each step as:
        STEP: [number]
        ACTION: [search/save]
        QUERY: [search query or memory key]
        VALUE: [if saving, what to save]"""
        
        response = self.generate_response(messages, prompt)
        
        # Parse steps
        steps = []
        current_step = {}
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('STEP:'):
                if current_step:
                    steps.append(current_step)
                current_step = {}
            elif line.startswith('ACTION:'):
                current_step['action'] = line.replace('ACTION:', '').strip().lower()
            elif line.startswith('QUERY:') or line.startswith('KEY:'):
                current_step['query'] = line.split(':', 1)[1].strip()
                current_step['key'] = current_step['query']
            elif line.startswith('VALUE:'):
                current_step['value'] = line.replace('VALUE:', '').strip()
        
        if current_step:
            steps.append(current_step)
            
        # Default if parsing fails
        if not steps:
            steps = [
                {'action': 'search', 'query': instruction},
                {'action': 'save', 'key': 'main_topic', 'value': instruction}
            ]
            
        return steps
    
    def generate_research_summary(self, findings: List[Dict]) -> str:
        """Generate a summary of research findings."""
        prompt = "Summarize the key findings from the research:\n\n"
        
        for finding in findings:
            prompt += f"Query: {finding['query']}\n"
            prompt += f"Result: {finding['result'][:500]}...\n\n"
        
        prompt += "Provide a concise summary highlighting the most important information."
        
        return self.generate_response([], prompt)