from typing import List, Dict, Any, Optional
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.base_agent import BaseAgent
from src.tools import RESEARCH_TOOLS

class SwarmResearchAgent(BaseAgent):
    """Research agent for the swarm team that can communicate directly with other agents."""
    
    def __init__(self):
        super().__init__("Research Agent (Swarm Team)", RESEARCH_TOOLS)
        self.research_phase = "initial"
        self.research_iterations = 0
        self.max_iterations = 10  # Increased from 5 to allow more thorough research
    
    def get_system_prompt(self) -> str:
        return """You are a Research Agent in a decentralized swarm team.
        
        Your responsibilities:
        1. Create a detailed outline for the report (Abstract + Introduction + 3+ body sections + Conclusion)
        2. Search for relevant and credible information to fill each section
        3. Extract and validate key facts and data points
        4. Save findings to shared memory organized by section
        5. Ensure sufficient research for EACH section of the outline
        6. Be demanding - if you don't have enough info, keep searching
        7. Communicate findings to the Analysis agent with clear structure
        
        You work autonomously and collaboratively. Use your tools strategically:
        - First create an outline with create_outline tool (request 3+ body sections)
        - Use search tools extensively to find detailed information
        - Use extraction tools to process findings
        - Use memory tools to save data organized by section
        - Use validation tools to check facts
        
        DO NOT proceed to Analysis until you have substantial research for ALL sections.
        Be aggressive about gathering comprehensive information."""
    
    def process(self, messages: List[AnyMessage], from_agent: Optional[str] = None, 
                instruction: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Process research task using dynamic tool selection."""
        
        # Update phase based on context
        if from_agent == "analysis" and instruction:
            self.research_phase = "supplementary"
        elif from_agent == "writer" and instruction:
            self.research_phase = "filling_gaps"
        
        # First, create an outline if we haven't yet
        if self.research_phase == "initial":
            outline = self.create_research_outline(messages)
            if outline:
                # Save outline to memory
                for tool in self.tools:
                    if 'save' in tool.name.lower() and 'memory' in tool.name.lower():
                        self.execute_tool(tool.name, f"report_outline::{outline}")
                        break
        
        # Main research loop
        research_complete = False
        all_findings = []
        sections_researched = set()
        
        while not research_complete and self.research_iterations < self.max_iterations:
            # Plan next action
            context = f"Research phase: {self.research_phase}, Iteration: {self.research_iterations + 1}"
            context += f", Sections researched: {len(sections_researched)}/5"
            if instruction:
                context += f", Specific request: {instruction}"
            
            action_plan = self.plan_next_action(messages, context)
            
            # Execute the planned action
            if action_plan['action'] == 'respond':
                # Agent decided to respond directly
                response_content = action_plan['input']
                all_findings.append(response_content)
            else:
                # Execute tool
                result = self.execute_tool(action_plan['action'], action_plan['input'])
                all_findings.append({
                    'tool': action_plan['action'],
                    'result': result,
                    'reasoning': action_plan['reasoning']
                })
                
                # Track which section this research covers
                if 'section' in action_plan['reasoning'].lower():
                    # Extract section number/name
                    sections_researched.add(len(all_findings))
                
                # Add result to messages for context
                messages.append(AIMessage(content=f"Tool {action_plan['action']} result: {result[:200]}..."))
            
            self.research_iterations += 1
            
            # Check if research is complete - be more demanding
            research_complete = self.assess_research_completeness(
                all_findings, messages, sections_researched
            )
        
        # Prepare final response
        return self.prepare_final_response(all_findings, from_agent)
    
    def create_research_outline(self, messages: List[AnyMessage]) -> str:
        """Create initial research outline."""
        # Check if create_outline tool is available
        outline_tool = None
        for tool in self.tools:
            if 'outline' in tool.name.lower():
                outline_tool = tool
                break
        
        if outline_tool:
            # Extract topic and word count from messages
            topic = ""
            word_count = 500  # default
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    content = msg.content
                    topic = content
                    # Extract word count if specified
                    import re
                    match = re.search(r'(\d+)[\s-]?word', content)
                    if match:
                        word_count = int(match.group(1))
                    break
            
            # Determine body sections based on word count
            if word_count <= 500:
                body_sections = 3
            elif word_count <= 1000:
                body_sections = 4
            elif word_count <= 1500:
                body_sections = 5
            else:
                body_sections = 6
            
            # Create outline with appropriate number of body sections
            outline = self.execute_tool(outline_tool.name, f"{topic}::{body_sections}")
            self.log(f"Created report outline with Abstract + Introduction + {body_sections} body sections + Conclusion")
            return outline
        else:
            # Create manual outline
            self.log("No outline tool available, creating manual outline")
            return """1. Abstract
2. Introduction
3. Current State and Key Trends
4. Challenges and Opportunities
5. Future Outlook and Implications
6. Conclusion and Recommendations"""
    
    def assess_research_completeness(self, findings: List, messages: List[AnyMessage], 
                                   sections_researched: set) -> bool:
        """Dynamically assess if research is complete - be more demanding."""
        # Require at least 8 findings (more than before)
        if len(findings) < 8:
            return False
        
        # Require coverage of at least 5 distinct topics/sections 
        # (Abstract, Intro, 3 body sections minimum, Conclusion)
        if len(sections_researched) < 5:
            return False
        
        if self.research_iterations >= self.max_iterations:
            return True
        
        # Ask the agent to self-assess with higher standards
        assessment_prompt = f"""Based on the research conducted so far:
        Number of findings: {len(findings)}
        Research phase: {self.research_phase}
        Sections covered: {len(sections_researched)}
        
        Determine if the research is sufficient for a COMPREHENSIVE report.
        We need:
        - Information for Abstract (summary of findings)
        - Context for Introduction
        - Detailed information for at least 3 body sections
        - Data to support Conclusion and recommendations
        - Multiple sources and perspectives
        - Recent data and statistics
        - Both challenges and opportunities
        
        Only answer COMPLETE if you have thorough research for ALL sections.
        
        Answer with: COMPLETE or CONTINUE"""
        
        response = self.generate_response(messages, assessment_prompt)
        return "COMPLETE" in response.upper()
    
    def prepare_final_response(self, findings: List, from_agent: Optional[str]) -> Dict[str, Any]:
        """Prepare the final response based on research findings."""
        # Synthesize findings
        synthesis_prompt = f"""Synthesize these research findings into a comprehensive summary:
        
        {self.format_findings(findings)}
        
        Create a well-structured synthesis that:
        1. Highlights main themes and patterns
        2. Identifies key data points
        3. Notes any gaps or limitations
        4. Suggests areas for analysis"""
        
        synthesis = self.generate_response([], synthesis_prompt)
        
        # Use memory tool to save synthesis (if available)
        memory_saved = False
        for tool in self.tools:
            if 'memory' in tool.name.lower() and 'save' in tool.name.lower():
                self.execute_tool(tool.name, f"research_synthesis::{synthesis}")
                memory_saved = True
                break
        
        # Prepare instructions for next agent
        if self.research_phase == "supplementary":
            next_agent = from_agent or "analysis"
            instruction = "Additional research completed. Please incorporate into your analysis."
        else:
            next_agent = "analysis"
            instruction = self.create_analysis_instruction(synthesis)
        
        # Reset for next time
        self.research_iterations = 0
        self.research_phase = "initial"
        
        return {
            "next_agent": next_agent,
            "complete": True,
            "instruction_for_next": instruction,
            "messages": [AIMessage(content=f"Research complete. Key findings:\n{synthesis[:500]}...\n\nHanding off to {next_agent} agent.")]
        }
    
    def format_findings(self, findings: List) -> str:
        """Format findings for synthesis."""
        formatted = []
        for i, finding in enumerate(findings, 1):
            if isinstance(finding, dict):
                formatted.append(f"Finding {i} ({finding.get('tool', 'Unknown tool')}):\n{finding.get('result', '')[:300]}...")
            else:
                formatted.append(f"Finding {i}: {str(finding)[:300]}...")
        return "\n\n".join(formatted)
    
    def create_analysis_instruction(self, synthesis: str) -> str:
        """Create instructions for the Analysis agent."""
        instruction_prompt = f"""Based on this research synthesis:
        {synthesis[:500]}...
        
        Create clear, actionable instructions for the Analysis agent.
        Include:
        1. Key areas to analyze
        2. Patterns to investigate
        3. Comparisons to make
        4. Insights to derive
        
        Keep instructions concise and specific."""
        
        return self.generate_response([], instruction_prompt)