from typing import List, Dict, Any
from langchain_core.messages import AnyMessage, AIMessage
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.base_agent import BaseAgent
from src.tools import WRITER_TOOLS

class WriterAgent(BaseAgent):
    """Writer agent for the orchestrator team."""
    
    def __init__(self):
        super().__init__("Writer Agent (Orchestrator Team)", WRITER_TOOLS)
        self.target_word_count = 500  # Default
    
    def get_system_prompt(self) -> str:
        return """You are a Writer Agent working under an Orchestrator's guidance.
        
        Your responsibilities:
        1. Read research findings and analysis from shared memory
        2. Create well-structured, coherent reports
        3. Maintain consistent tone and style throughout
        4. Ensure proper flow and logical progression
        5. Meet specified word count requirements
        6. Include all key insights and conclusions
        
        Focus on clarity, readability, and comprehensive coverage."""
    
    def process(self, messages: List[AnyMessage], instruction: str = None, **kwargs) -> Dict[str, Any]:
        """Create final report based on research and analysis."""
        # Set target word count if specified
        if 'word_count' in kwargs:
            self.target_word_count = kwargs['word_count']
        
        # Read all relevant data from memory
        research_summary = self.execute_tool('read_from_memory', 'research_summary')
        analysis_results = self.execute_tool('read_from_memory', 'analysis_results')
        key_insights = self.execute_tool('read_from_memory', 'key_insights')
        
        # Create outline
        outline = self.create_report_outline(instruction, key_insights)
        
        # Write each section
        sections = self.write_sections(outline, research_summary, analysis_results, key_insights)
        
        # Combine into final report
        report = self.combine_sections(sections)
        
        # Check word count and adjust if needed
        current_count = int(self.execute_tool('word_count', report).split(': ')[1])
        if abs(current_count - self.target_word_count) > 50:
            report = self.adjust_length(report, self.target_word_count)
        
        # Final quality check
        quality_score = self.assess_quality(report)
        
        # Save final report
        self.execute_tool('save_to_memory', f"final_report::{report}")
        
        return {
            "complete": True,
            "messages": [AIMessage(content=f"Report completed. Quality score: {quality_score}/10")],
            "report": report,
            "word_count": self.execute_tool('word_count', report)
        }
    
    def create_report_outline(self, instruction: str, insights: str) -> Dict[str, str]:
        """Create a structured outline for the report."""
        prompt = f"""Create a report outline based on:
        Instruction: {instruction}
        Key insights: {insights}
        
        The outline should include:
        - Introduction
        - 2-4 main sections
        - Conclusion
        
        For each section, provide a brief description of what it should contain.
        
        Format as:
        SECTION: [name]
        CONTENT: [description]"""
        
        response = self.generate_response([], prompt)
        
        # Parse outline
        outline = {}
        current_section = None
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('SECTION:'):
                current_section = line.replace('SECTION:', '').strip()
            elif line.startswith('CONTENT:') and current_section:
                outline[current_section] = line.replace('CONTENT:', '').strip()
        
        # Default outline if parsing fails
        if not outline:
            outline = {
                "Introduction": "Overview of the topic and report objectives",
                "Main Findings": "Key research findings and data",
                "Analysis": "In-depth analysis and insights",
                "Conclusion": "Summary and final thoughts"
            }
            
        return outline
    
    def write_sections(self, outline: Dict[str, str], research: str, analysis: str, insights: str) -> Dict[str, str]:
        """Write each section of the report."""
        sections = {}
        
        for section_name, section_desc in outline.items():
            prompt = f"""Write the '{section_name}' section of the report.
            
            Section description: {section_desc}
            
            Available information:
            Research: {research[:500]}...
            Analysis: {analysis[:500]}...
            Insights: {insights}
            
            Write in a clear, professional tone. Make it engaging and informative."""
            
            sections[section_name] = self.generate_response([], prompt)
            
        return sections
    
    def combine_sections(self, sections: Dict[str, str]) -> str:
        """Combine sections into a cohesive report."""
        report_parts = []
        
        for section_name, content in sections.items():
            # Add section header
            report_parts.append(f"## {section_name}\n")
            report_parts.append(content)
            report_parts.append("\n")
        
        return '\n'.join(report_parts)
    
    def adjust_length(self, report: str, target_count: int) -> str:
        """Adjust report length to meet word count requirement."""
        current_count = int(self.execute_tool('word_count', report).split(': ')[1])
        
        if current_count < target_count:
            prompt = f"""The current report is {current_count} words. 
            Expand it to approximately {target_count} words by adding more detail and examples.
            
            Current report:
            {report}"""
        else:
            prompt = f"""The current report is {current_count} words. 
            Condense it to approximately {target_count} words while keeping all key information.
            
            Current report:
            {report}"""
            
        return self.generate_response([], prompt)
    
    def assess_quality(self, report: str) -> int:
        """Assess the quality of the final report."""
        prompt = f"""Rate the following report on a scale of 1-10 based on:
        - Clarity and coherence
        - Comprehensive coverage
        - Logical flow
        - Professional tone
        - Factual accuracy
        
        Report:
        {report[:1000]}...
        
        Provide just the numeric score."""
        
        response = self.generate_response([], prompt)
        
        try:
            return int(response.strip())
        except:
            return 7  # Default score