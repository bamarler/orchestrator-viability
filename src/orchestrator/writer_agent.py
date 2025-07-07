from typing import List, Dict, Any, Optional
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.base_agent import BaseAgent
from src.tools import WRITER_TOOLS

class WriterAgent(BaseAgent):
    """Writer agent for the orchestrator team using dynamic tool selection."""
    
    def __init__(self):
        super().__init__("Writer Agent (Orchestrator Team)", WRITER_TOOLS)
        self.target_word_count = 500  # Default, will be updated
        self.minimum_word_count = 500  # Will match target
        self.writing_iterations = 0
        self.max_iterations = 5
        self.min_quality_score = 7
    
    def get_system_prompt(self) -> str:
        return """You are a Writer Agent working under an Orchestrator's guidance.
        
        Your responsibilities:
        1. Read ALL research findings and analysis from shared memory
        2. Create COMPLETE, well-structured reports (NOT outlines!)
        3. Follow proper structure: Abstract + Introduction + Body Sections + Conclusion
        4. Write substantive content for each section with facts and data
        5. Maintain consistent tone and professional quality
        6. Meet specified word count requirements
        7. Include all key insights and supporting evidence
        
        Use your tools strategically:
        - Use read_from_memory to access all research and analysis
        - Use create_outline to structure the report properly
        - Use word_count to track progress
        - Use save_to_memory to save the final report
        
        CRITICAL: Write ACTUAL CONTENT, never submit just an outline or structure.
        Each section must contain real information, analysis, and insights.
        Only declare complete when you have a full, professional-quality report."""
    
    def process(self, messages: List[AnyMessage], instruction: str = None, **kwargs) -> Dict[str, Any]:
        """Create final report using dynamic tool selection and writing."""
        self.extract_word_count_from_messages(messages)
        
        if instruction:
            self.extract_word_count_from_instruction(instruction)
        
        report_sections = []
        current_report = ""
        
        while self.writing_iterations < self.max_iterations:
            context = self.build_writing_context(instruction, current_report, report_sections)
            
            action_plan = self.plan_next_action(messages, context)
            
            if action_plan['action'] == 'respond':
                report_sections.append(action_plan['input'])
            else:
                result = self.execute_tool(action_plan['action'], action_plan['input'])
                
                if 'word_count' in action_plan['action'].lower():
                    self.log(f"Current word count: {result}")
                elif 'memory' in action_plan['action'].lower() and 'read' in action_plan['action'].lower():
                    messages.append(AIMessage(content=f"Retrieved data: {result[:200]}..."))
                elif 'outline' in action_plan['action'].lower():
                    report_sections.append(f"Outline created: {result}")
                else:
                    messages.append(AIMessage(content=f"Tool result: {result[:100]}..."))
            
            self.writing_iterations += 1
            
            current_report = self.compile_full_report(report_sections)
            
            if self.is_report_complete(current_report):
                break
        
        final_report = self.finalize_report(current_report)
        quality_score = self.assess_report_quality(final_report)
        word_count = self.get_word_count(final_report)
        
        self.save_final_report(final_report)
        
        self.writing_iterations = 0
        
        return {
            "complete": True,
            "messages": [AIMessage(content=f"Report completed!\n- Quality score: {quality_score}/10\n- Word count: {word_count}\n- Structure: Complete with all sections")],
            "report": final_report,
            "word_count": word_count
        }
    
    def extract_word_count_from_messages(self, messages: List[AnyMessage]):
        """Extract target word count from messages."""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                content = msg.content
                match = re.search(r'(\d+)[\s-]?word', content, re.IGNORECASE)
                if match:
                    self.target_word_count = int(match.group(1))
                    self.minimum_word_count = self.target_word_count
                    self.log(f"Target word count set to: {self.target_word_count}")
                    break
    
    def extract_word_count_from_instruction(self, instruction: str):
        """Extract target word count from instruction."""
        if instruction:
            match = re.search(r'(\d+)[\s-]?word', instruction, re.IGNORECASE)
            if match:
                self.target_word_count = int(match.group(1))
                self.minimum_word_count = self.target_word_count
                self.log(f"Target word count set to: {self.target_word_count}")
    
    def build_writing_context(self, instruction: str, current_report: str, sections: List) -> str:
        """Build context for planning next writing action."""
        # Check available data
        available_data = self.check_available_data()
        
        # Get current word count
        current_words = len(current_report.split()) if current_report else 0
        
        context_parts = [
            f"Instruction: {instruction}" if instruction else "Write comprehensive report",
            f"Target: {self.target_word_count} words",
            f"Current: {current_words} words",
            f"Sections written: {len(sections)}",
            f"Available data: {available_data}",
            f"Iteration: {self.writing_iterations + 1}/{self.max_iterations}"
        ]
        
        return " | ".join(context_parts)
    
    def check_available_data(self) -> str:
        """Check what data is available for writing."""
        available = []
        
        # Key data to check for
        important_keys = [
            'research_synthesis', 'research_summary',
            'analysis_insights', 'key_insights',
            'report_outline', 'consistency_report'
        ]
        
        # Use list_memory_keys if available
        for tool in self.tools:
            if 'list' in tool.name.lower() and 'memory' in tool.name.lower():
                memory_keys = self.execute_tool(tool.name, '')
                for key in important_keys:
                    if key in memory_keys:
                        available.append(key.replace('_', ' '))
                break
        
        return ", ".join(available) if available else "Limited data available"
    
    def compile_full_report(self, sections: List[str]) -> str:
        """Compile sections into a complete report."""
        # Filter out meta-information
        content_sections = []
        for section in sections:
            if not section.startswith("Outline created:") and \
               not section.startswith("Retrieved data:") and \
               len(section) > 50:  # Real content
                content_sections.append(section)
        
        # If we don't have enough content, write it now
        if len(content_sections) < 3 or sum(len(s.split()) for s in content_sections) < self.target_word_count * 0.7:
            return self.write_complete_report()
        
        return "\n\n".join(content_sections)
    
    def write_complete_report(self) -> str:
        """Write a complete report using all available data."""
        # Gather all necessary data
        research_data = ""
        analysis_data = ""
        outline_data = ""
        
        for tool in self.tools:
            if 'read' in tool.name.lower() and 'memory' in tool.name.lower():
                research_data = self.execute_tool(tool.name, 'research_synthesis')
                analysis_data = self.execute_tool(tool.name, 'analysis_insights')
                outline_data = self.execute_tool(tool.name, 'report_outline')
                break
        
        # Determine report structure
        num_body_sections = self.determine_body_sections()
        
        # Calculate word distribution
        abstract_words = int(self.target_word_count * 0.15)
        intro_words = int(self.target_word_count * 0.10)
        conclusion_words = int(self.target_word_count * 0.10)
        body_total = self.target_word_count - abstract_words - intro_words - conclusion_words
        words_per_body = body_total // num_body_sections
        
        # Extract section topics from outline
        section_topics = self.extract_section_topics(outline_data, num_body_sections)
        
        # Write the complete report
        write_prompt = f"""Write a COMPLETE professional report with the following structure:
        
        1. Abstract ({abstract_words} words)
           - Executive summary of the entire report
           - Key findings and conclusions
           
        2. Introduction ({intro_words} words)
           - Background and context
           - Purpose and scope of the report
        """
        
        # Add body sections
        for i, topic in enumerate(section_topics, 1):
            write_prompt += f"""
        {i + 2}. {topic} ({words_per_body} words)
           - Detailed analysis with specific data
           - Examples and evidence from research
           - Clear insights and implications
        """
        
        write_prompt += f"""
        {num_body_sections + 3}. Conclusion ({conclusion_words} words)
           - Summary of key findings
           - Actionable recommendations
           - Future outlook
        
        Total target: {self.target_word_count} words
        
        Use this data:
        Research findings: {research_data}
        Analysis insights: {analysis_data}
        
        CRITICAL: Write ACTUAL CONTENT with real information, not placeholders or outlines!
        Each section must contain substantive analysis and specific details."""
        
        return self.generate_response([], write_prompt)
    
    def determine_body_sections(self) -> int:
        """Determine number of body sections based on word count."""
        if self.target_word_count <= 500:
            return 3
        elif self.target_word_count <= 1000:
            return 4
        elif self.target_word_count <= 1500:
            return 5
        elif self.target_word_count <= 2000:
            return 6
        else:
            return 6 + ((self.target_word_count - 2000) // 500)
    
    def extract_section_topics(self, outline: str, num_sections: int) -> List[str]:
        """Extract section topics from outline or generate defaults."""
        default_topics = [
            "Current State and Overview",
            "Key Trends and Developments",
            "Challenges and Opportunities",
            "Analysis and Implications",
            "Case Studies and Examples",
            "Future Outlook and Predictions",
            "Implementation Strategies",
            "Best Practices and Recommendations"
        ]
        
        if "No data found" in outline:
            return default_topics[:num_sections]
        
        # Try to extract from outline
        topics = []
        lines = outline.split('\n')
        for line in lines:
            # Skip meta lines
            if any(skip in line.lower() for skip in ['abstract', 'introduction', 'conclusion', 'outline']):
                continue
            # Extract numbered sections
            if re.match(r'\d+\.', line.strip()):
                topic = re.sub(r'^\d+\.\s*', '', line.strip())
                if topic and '[' not in topic:  # Skip placeholders
                    topics.append(topic)
        
        # Ensure we have enough topics
        while len(topics) < num_sections:
            if len(topics) < len(default_topics):
                topics.append(default_topics[len(topics)])
            else:
                topics.append(f"Additional Analysis {len(topics) - len(default_topics) + 1}")
        
        return topics[:num_sections]
    
    def is_report_complete(self, report: str) -> bool:
        """Check if the report is complete and meets standards."""
        if not report:
            return False
        
        word_count = len(report.split())
        
        # Never complete if below minimum
        if word_count < self.minimum_word_count * 0.9:  # Allow 10% tolerance
            self.log(f"Report has {word_count} words, need at least {int(self.minimum_word_count * 0.9)}")
            return False
        
        # Check if it's just an outline
        if any(marker in report.lower() for marker in ['[section', 'content would be', 'to be added']):
            self.log("Report appears to be an outline, not actual content")
            return False
        
        # Check structure
        has_abstract = 'abstract' in report.lower()
        has_intro = 'introduction' in report.lower()
        has_conclusion = 'conclusion' in report.lower()
        
        if not (has_abstract and has_intro and has_conclusion):
            self.log("Report missing required sections")
            return False
        
        # Passed basic checks
        return True
    
    def finalize_report(self, report: str) -> str:
        """Finalize the report with any last improvements."""
        # Quick quality check
        if "outline" in report.lower()[:200] or len(report.split()) < self.minimum_word_count:
            # Emergency rewrite
            self.log("Report needs complete rewrite")
            return self.write_complete_report()
        
        return report
    
    def assess_report_quality(self, report: str) -> int:
        """Assess the quality of the report."""
        # Check for outline markers
        if any(marker in report.lower() for marker in ['[section', 'content would be', 'to be added']):
            return 2  # Very low score for outlines
        
        quality_prompt = f"""Rate this report from 1-10 based on:
        1. Complete content (not just outline)
        2. All sections properly developed
        3. Professional writing quality
        4. Logical flow and coherence
        5. Use of specific data and examples
        6. Actionable insights and conclusions
        
        Report preview: {report}...
        
        Give ONLY a number 1-10. Be strict - only 7+ for truly complete, professional reports."""
        
        response = self.generate_response([], quality_prompt)
        
        try:
            score = int(re.search(r'\d+', response).group())
            return min(max(score, 1), 10)
        except:
            return 5
    
    def get_word_count(self, report: str) -> int:
        """Get word count of the report."""
        for tool in self.tools:
            if 'word_count' in tool.name.lower():
                result = self.execute_tool(tool.name, report)
                try:
                    return int(re.search(r'\d+', result).group())
                except:
                    pass
        
        return len(report.split())
    
    def save_final_report(self, report: str):
        """Save the final report to memory."""
        for tool in self.tools:
            if 'save' in tool.name.lower() and 'memory' in tool.name.lower():
                self.execute_tool(tool.name, f"final_report::{report}")
                self.log("Final report saved to memory")
                break