from typing import List, Dict, Any
from langchain_core.messages import AnyMessage, AIMessage
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.base_agent import BaseAgent
from src.tools import ANALYSIS_TOOLS

class AnalysisAgent(BaseAgent):
    """Analysis agent for the orchestrator team."""
    
    def __init__(self):
        super().__init__("Analysis Agent (Orchestrator Team)", ANALYSIS_TOOLS)
        self.analysis_complete = False
    
    def get_system_prompt(self) -> str:
        return """You are an Analysis Agent working under an Orchestrator's guidance.
        
        Your responsibilities:
        1. Read and synthesize research findings from shared memory
        2. Identify patterns, trends, and key insights
        3. Perform calculations and comparisons when needed
        4. Check for factual consistency across sources
        5. Draw meaningful conclusions from the data
        6. Prepare structured analysis for the Writer agent
        
        Focus on providing actionable insights and clear conclusions."""
    
    def process(self, messages: List[AnyMessage], instruction: str = None, **kwargs) -> Dict[str, Any]:
        """Analyze research findings based on orchestrator's instruction."""
        # Read available research data
        memory_keys = self.execute_tool('list_memory_keys', '')
        research_summary = self.execute_tool('read_from_memory', 'research_summary')
        
        # Perform analysis
        analysis_results = self.perform_analysis(messages, instruction, research_summary)
        
        # Check for consistency
        consistency_checks = self.check_consistency(analysis_results)
        
        # Generate insights
        insights = self.generate_insights(analysis_results, consistency_checks)
        
        # Save analysis to memory
        self.execute_tool('save_to_memory', f'analysis_results::{analysis_results}')
        self.execute_tool('save_to_memory', f'insights::{insights}')
        
        # Prepare summary for orchestrator
        summary = self.generate_analysis_summary(analysis_results, insights)
        
        return {
            "complete": True,
            "messages": [AIMessage(content=f"Analysis completed. Summary:\n{summary}")],
            "analysis": analysis_results,
            "insights": insights
        }
    
    def perform_analysis(self, messages: List[AnyMessage], instruction: str, research_data: str) -> Dict[str, Any]:
        """Perform detailed analysis of research data."""
        prompt = f"""Given the instruction: {instruction}
        
        And the research data:
        {research_data}
        
        Perform a comprehensive analysis covering:
        1. Key findings and their significance
        2. Patterns or trends identified
        3. Data points requiring calculation or comparison
        4. Potential gaps or areas needing clarification
        
        Structure your analysis with clear sections."""
        
        response = self.generate_response(messages, prompt)
        
        # Parse into structured format
        sections = {}
        current_section = "general"
        current_content = []
        
        for line in response.split('\n'):
            if line.strip() and line.strip().endswith(':'):
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = line.strip()[:-1].lower().replace(' ', '_')
                current_content = []
            else:
                current_content.append(line)
        
        if current_content:
            sections[current_section] = '\n'.join(current_content)
            
        return sections
    
    def check_consistency(self, analysis_results: Dict[str, Any]) -> List[Dict]:
        """Check for consistency across different findings."""
        checks = []
        
        # Extract claims from analysis
        all_text = ' '.join(str(v) for v in analysis_results.values())
        sentences = all_text.split('.')
        
        # Compare key sentences for consistency
        for i in range(len(sentences)-1):
            for j in range(i+1, min(i+3, len(sentences))):
                if len(sentences[i].strip()) > 20 and len(sentences[j].strip()) > 20:
                    result = self.execute_tool('check_factual_consistency', 
                                             f"{sentences[i]}::{sentences[j]}")
                    if "inconsistent" in result.lower():
                        checks.append({
                            "claim1": sentences[i].strip(),
                            "claim2": sentences[j].strip(),
                            "result": result
                        })
        
        return checks
    
    def generate_insights(self, analysis_results: Dict, consistency_checks: List[Dict]) -> str:
        """Generate key insights from analysis."""
        prompt = f"""Based on the analysis results:
        {analysis_results}
        
        And consistency checks:
        {consistency_checks}
        
        Generate 3-5 key insights that would be valuable for the final report.
        Each insight should be:
        - Actionable and specific
        - Supported by the data
        - Relevant to the original task
        
        Format as numbered list."""
        
        return self.generate_response([], prompt)
    
    def generate_analysis_summary(self, analysis_results: Dict, insights: str) -> str:
        """Generate summary for the orchestrator."""
        prompt = f"""Provide a concise summary of the analysis including:
        1. Main analytical findings
        2. Key insights discovered
        3. Any areas of concern or inconsistency
        4. Recommendations for the final report
        
        Keep it brief but comprehensive."""
        
        return self.generate_response([], prompt)