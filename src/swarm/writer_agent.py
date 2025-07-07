from typing import List, Dict, Any, Optional
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.base_agent import BaseAgent
from src.tools import WRITER_TOOLS

class SwarmWriterAgent(BaseAgent):
    """Writer agent for the swarm team that determines when the task is complete."""
    
    def __init__(self):
        super().__init__("Writer Agent (Swarm Team)", WRITER_TOOLS)
        self.target_word_count = 500  # Will be updated from messages
        self.minimum_word_count = 500  # Will be updated to match target
        self.revision_count = 0
        self.max_revisions = 3
        self.writing_phase = "drafting"
        self.min_body_sections = 3  # Minimum number of body sections
    
    def get_system_prompt(self) -> str:
        return """You are a Writer Agent in a decentralized swarm team.
        
        Your responsibilities:
        1. Create COMPLETE reports with ACTUAL CONTENT (not just outlines!)
        2. Follow the structure: Abstract + Introduction + (minimum 3 body sections) + Conclusion
        3. Write detailed, substantive content for each section
        4. Meet word count requirements (will be specified in the task)
        5. Ensure high quality with actionable insights
        6. Be demanding - if you lack info, request it from other agents
        7. NEVER submit just an outline - always write full content
        
        Report structure:
        - Abstract: Executive summary of the entire report
        - Introduction: Background, context, and scope
        - Body Sections (3+): Main content with analysis, data, examples
        - Conclusion: Summary, implications, and recommendations
        
        You work autonomously and can:
        - Use memory tools to access ALL research and analysis
        - Use writing/formatting tools to structure content
        - Use quality assessment tools to evaluate your work
        - Use word count tools to track length
        - Request MORE DATA from other agents if sections are thin
        - Revise until quality score is 8+ out of 10
        - Only declare complete when report is FULLY WRITTEN
        
        Remember: You must WRITE the report, not just plan it!"""
    
    def process(self, messages: List[AnyMessage], from_agent: Optional[str] = None,
                instruction: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Process writing task using dynamic tool selection."""
        
        # Extract word count requirement
        self.extract_word_count(messages)
        
        # Initialize report content
        report_sections = []
        current_report = ""
        
        # Main writing loop
        writing_complete = False
        iterations = 0
        max_iterations = 8
        
        while not writing_complete and iterations < max_iterations:
            # Build context for planning
            context = self.build_writing_context(
                from_agent, instruction, current_report, report_sections
            )
            
            # Plan next writing action
            action_plan = self.plan_next_action(messages, context)
            
            # Execute planned action
            if action_plan['action'] == 'respond':
                # Direct writing
                report_sections.append(action_plan['input'])
            else:
                # Use a tool
                result = self.execute_tool(action_plan['action'], action_plan['input'])
                
                # Handle tool results
                if 'word_count' in action_plan['action'].lower():
                    # Track word count
                    self.update_memory('last_word_count', result)
                elif 'memory' in action_plan['action'].lower() and 'read' in action_plan['action'].lower():
                    # Retrieved data, use it for writing
                    messages.append(AIMessage(content=f"Retrieved: {result[:200]}..."))
                elif 'quality' in action_plan['action'].lower() or 'assess' in action_plan['action'].lower():
                    # Quality assessment
                    self.update_memory('quality_assessment', result)
                else:
                    # Other tool results
                    report_sections.append(f"[Tool result: {result[:100]}...]")
            
            iterations += 1
            
            # Compile current report
            current_report = self.compile_report(report_sections)
            
            # Check if writing is complete
            writing_complete = self.assess_completion(current_report, iterations)
        
        # Finalize and return
        return self.finalize_report(current_report, from_agent)
    
    def determine_body_sections(self) -> int:
        """Determine number of body sections based on word count."""
        # Scale body sections with word count
        if self.target_word_count <= 500:
            return 3  # Minimum
        elif self.target_word_count <= 1000:
            return 4
        elif self.target_word_count <= 1500:
            return 5
        elif self.target_word_count <= 2000:
            return 6
        else:
            # Add 1 section per 500 words above 2000
            extra_sections = (self.target_word_count - 2000) // 500
            return 6 + extra_sections
    
    def extract_word_count(self, messages: List[AnyMessage]):
        """Extract target word count from messages."""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                content = msg.content
                match = re.search(r'(\d+)[\s-]?word', content, re.IGNORECASE)
                if match:
                    self.target_word_count = int(match.group(1))
                    self.minimum_word_count = self.target_word_count  # Match the target
                    self.log(f"Target word count set to: {self.target_word_count}")
    
    def build_writing_context(self, from_agent: Optional[str], 
                            instruction: Optional[str],
                            current_report: str,
                            sections: List) -> str:
        """Build context for action planning."""
        context_parts = [
            f"Writing phase: {self.writing_phase}",
            f"Target: {self.target_word_count} words",
            f"Current sections: {len(sections)}",
            f"Revision: {self.revision_count}/{self.max_revisions}"
        ]
        
        if from_agent:
            context_parts.append(f"Input from: {from_agent}")
        
        if instruction:
            context_parts.append(f"Instructions: {instruction[:100]}...")
        
        if current_report:
            # Check word count if we have content
            word_count = len(current_report.split())
            context_parts.append(f"Current length: {word_count} words")
        
        # Check available data
        data_check = self.check_available_data()
        context_parts.append(f"Available data: {data_check}")
        
        return " | ".join(context_parts)
    
    def check_available_data(self) -> str:
        """Check what data is available for writing."""
        available = []
        
        # Quick check for common memory keys
        memory_check_keys = ['analysis_insights', 'research_synthesis', 'key_insights']
        
        for tool in self.tools:
            if 'memory' in tool.name.lower() and 'read' in tool.name.lower():
                for key in memory_check_keys:
                    result = self.execute_tool(tool.name, key)
                    if "No data found" not in result:
                        available.append(key.replace('_', ' '))
                break
        
        return ", ".join(available) if available else "No data found"
    
    def compile_report(self, sections: List[str]) -> str:
        """Compile report sections into a complete document."""
        # Remove tool result markers
        clean_sections = []
        for section in sections:
            if not section.startswith("[Tool result:"):
                clean_sections.append(section)
        
        # Join sections with appropriate spacing
        report = "\n\n".join(clean_sections)
        
        # If report is too short, we need to write actual content
        word_count = len(report.split())
        if word_count < (self.target_word_count * 0.8) and self.writing_phase == "drafting":
            # Get available data
            insights = ""
            research = ""
            outline = ""
            for tool in self.tools:
                if 'memory' in tool.name.lower() and 'read' in tool.name.lower():
                    insights = self.execute_tool(tool.name, 'analysis_insights')
                    research = self.execute_tool(tool.name, 'research_synthesis')
                    outline = self.execute_tool(tool.name, 'report_outline')
                    break
            
            # Calculate dynamic word distribution
            # Structure: Abstract + Intro + (dynamic body sections) + Conclusion
            num_body_sections = self.determine_body_sections()
            total_sections = 2 + num_body_sections + 1  # Abstract/Intro + Bodies + Conclusion
            
            # Word distribution (approximate)
            abstract_words = int(self.target_word_count * 0.15)  # 15% for abstract
            intro_words = int(self.target_word_count * 0.10)     # 10% for intro
            conclusion_words = int(self.target_word_count * 0.10) # 10% for conclusion
            
            # Remaining words for body sections
            body_total_words = self.target_word_count - abstract_words - intro_words - conclusion_words
            words_per_body = body_total_words // num_body_sections
            
            # Get body section topics from outline if available
            body_topics = self.extract_body_topics(outline)
            
            # Write actual content, not just structure
            content_prompt = f"""Write a COMPLETE report (not an outline!) with this structure:
            
            1. Abstract ({abstract_words} words)
               - Summarize the entire report
               - Key findings and conclusions
            
            2. Introduction ({intro_words} words)
               - Background and context
               - Purpose and scope
            """
            
            # Add body sections dynamically
            for i, topic in enumerate(body_topics[:num_body_sections], 1):
                content_prompt += f"""
            {i + 2}. {topic} ({words_per_body} words)
               - Detailed analysis with facts and data
               - Specific examples and evidence
            """
            
            content_prompt += f"""
            {num_body_sections + 3}. Conclusion ({conclusion_words} words)
               - Summary of key points
               - Recommendations and future outlook
            
            Total target: {self.target_word_count} words minimum
            
            Available data:
            Research: {research[:1000]}...
            Analysis: {insights[:1000]}...
            
            IMPORTANT: Write ACTUAL CONTENT with facts, data, and insights.
            Do NOT just provide an outline or structure!
            Each section must contain substantive information."""
            
            report = self.generate_response([], content_prompt)
        
        return report
    
    def assess_completion(self, report: str, iterations: int) -> bool:
        """Assess if the report is complete."""
        if not report:
            return False
        
        word_count = len(report.split())
        
        # Never complete if below minimum word count
        if word_count < self.minimum_word_count:
            self.log(f"Report has {word_count} words, need minimum {self.minimum_word_count}")
            return False
        
        # Check if it's just an outline
        if "content would be added" in report.lower() or "[section" in report.lower():
            self.log("Report appears to be just an outline, not actual content")
            return False
        
        # Use quality assessment if available
        quality_tool = None
        for tool in self.tools:
            if 'quality' in tool.name.lower() or 'assess' in tool.name.lower():
                quality_tool = tool
                break
        
        if quality_tool:
            assessment = self.execute_tool(quality_tool.name, report)
            # Simple check - if assessment mentions "complete" or gives high score
            if "complete" in assessment.lower() or "good" in assessment.lower():
                return True
        
        # Check word count
        within_range = abs(word_count - self.target_word_count) <= 50
        
        # Check if we've done enough iterations
        sufficient_iterations = iterations >= 4
        
        return within_range and sufficient_iterations and word_count >= self.minimum_word_count
    
    def finalize_report(self, report: str, from_agent: Optional[str]) -> Dict[str, Any]:
        """Finalize the report and determine next steps."""
        # Final quality check
        quality_score = self.final_quality_assessment(report)
        
        # Save to memory if possible
        for tool in self.tools:
            if 'memory' in tool.name.lower() and 'save' in tool.name.lower():
                self.execute_tool(tool.name, f"final_report::{report}")
                break
        
        # Determine if we need revision
        if quality_score < 7 and self.revision_count < self.max_revisions:
            self.revision_count += 1
            self.writing_phase = "revising"
            
            # Identify what needs improvement
            improvement_areas = self.identify_improvements(report, quality_score)
            
            # Decide who can help
            if "more analysis" in improvement_areas.lower():
                return {
                    "next_agent": "analysis",
                    "complete": False,
                    "instruction_for_next": improvement_areas,
                    "messages": [AIMessage(content=f"Report needs improvement (score: {quality_score}/10). Requesting additional analysis.")]
                }
            elif "more research" in improvement_areas.lower():
                return {
                    "next_agent": "research",
                    "complete": False,
                    "instruction_for_next": improvement_areas,
                    "messages": [AIMessage(content=f"Report needs improvement (score: {quality_score}/10). Requesting additional research.")]
                }
            else:
                # Self-revise
                return {
                    "next_agent": "writer",
                    "complete": False,
                    "messages": [AIMessage(content=f"Self-revising report (score: {quality_score}/10). Areas to improve: {improvement_areas}")]
                }
        else:
            # Report is complete
            word_count = len(report.split())
            return {
                "next_agent": None,
                "complete": True,
                "report": report,
                "messages": [AIMessage(content=f"Report completed!\n- Quality score: {quality_score}/10\n- Word count: {word_count}\n- Revisions: {self.revision_count}\n\nFinal Report:\n{report}")]
            }
    
    def final_quality_assessment(self, report: str) -> float:
        """Perform final quality assessment with high standards."""
        # First check if it's actually content or just an outline
        if "content would be added" in report.lower() or "[section" in report.lower():
            return 2  # Very low score for outlines
        
        assessment_prompt = f"""Assess this report on a scale of 1-10:
        
        {report}...
        
        Consider:
        1. Is this ACTUAL CONTENT or just an outline? (If outline, score 2 or less)
        2. Clarity and coherence of writing
        3. Comprehensive coverage of all sections
        4. Presence of specific facts, data, and examples
        5. Logical flow between sections
        6. Actionable insights and conclusions
        7. Professional tone and grammar
        8. Meeting word count requirements
        
        Be strict - only score 7+ if this is a complete, well-written report.
        
        Provide a single number score (1-10)."""
        
        response = self.generate_response([], assessment_prompt)
        
        # Extract score
        try:
            score = float(re.search(r'(\d+)', response).group(1))
            return min(max(score, 1), 10)  # Ensure between 1-10
        except:
            return 5  # Default to medium score
    
    def extract_body_topics(self, outline: str) -> List[str]:
        """Extract body section topics from outline."""
        default_topics = [
            "Current State and Key Trends",
            "Challenges and Opportunities", 
            "Analysis and Implications",
            "Case Studies and Examples",
            "Future Outlook and Predictions",
            "Implementation Strategies",
            "Economic and Social Impact",
            "Best Practices and Lessons Learned"
        ]
        
        if "No data found" in outline:
            return default_topics
        
        # Try to extract topics from outline
        topics = []
        lines = outline.split('\n')
        for line in lines:
            # Skip Abstract, Introduction, Conclusion lines
            if any(skip in line.lower() for skip in ['abstract', 'introduction', 'conclusion', 'report outline']):
                continue
            # Look for numbered sections
            if re.match(r'\d+\.', line.strip()):
                topic = re.sub(r'^\d+\.\s*', '', line.strip())
                if topic and topic != line.strip():
                    topics.append(topic)
        
        # Ensure we have enough topics for the required body sections
        required_sections = self.determine_body_sections()
        while len(topics) < required_sections:
            if len(topics) < len(default_topics):
                topics.append(default_topics[len(topics)])
            else:
                topics.append(f"Additional Analysis {len(topics) - len(default_topics) + 1}")
        
        return topics
    
    def identify_improvements(self, report: str, score: float) -> str:
        """Identify areas for improvement - be specific and demanding."""
        # Check if it's just an outline
        if "content would be added" in report.lower() or "[section" in report.lower():
            return "MORE RESEARCH needed - this is just an outline! Need actual content with facts, data, and analysis for all sections."
        
        improvement_prompt = f"""The report scored {score}/10. Identify specific improvements needed:
        
        Report excerpt: {report}...
        
        Check for:
        1. Is this actual content or just structure? (If structure, demand content!)
        2. Are all 5 main sections + abstract + conclusion present?
        3. Does each section have substantial content (100+ words)?
        4. Are there specific facts, statistics, and examples?
        5. Is the analysis deep enough?
        6. Are conclusions actionable?
        
        What SPECIFICALLY needs to be added or improved?
        If sections are missing or thin, which ones need:
        - MORE RESEARCH on specific topics?
        - MORE ANALYSIS of existing data?
        - Better structure/clarity?
        - Stronger conclusions?
        
        Be specific about what's needed and from whom (Research or Analysis agent)."""
        
        return self.generate_response([], improvement_prompt)