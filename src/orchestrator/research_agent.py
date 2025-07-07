from typing import List, Dict, Any, Optional
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.base_agent import BaseAgent
from src.tools import RESEARCH_TOOLS

class ResearchAgent(BaseAgent):
    """Research agent for the orchestrator team using dynamic tool selection."""
    
    def __init__(self):
        super().__init__("Research Agent (Orchestrator Team)", RESEARCH_TOOLS)
        self.research_iterations = 0
        self.max_iterations = 8
        self.target_word_count = 500  # Will be updated from instruction
        self.min_findings_required = 6
    
    def get_system_prompt(self) -> str:
        return """You are a Research Agent working under an Orchestrator's guidance.
        
        Your responsibilities:
        1. Create a comprehensive outline first (Abstract + Intro + 3+ body sections + Conclusion)
        2. Search extensively for relevant and credible information
        3. Extract and validate key facts and data points
        4. Save organized findings to shared memory by section
        5. Track sources for citation purposes
        6. Ensure sufficient research for EACH section of the outline
        
        Use your tools strategically:
        - First use create_outline tool to structure the research
        - Use web_search extensively with varied queries
        - Use extract_facts to identify key information
        - Use save_to_memory to organize findings by section
        - Use format_citation for source tracking
        
        Be thorough - don't stop until you have substantial information for ALL sections.
        Report back to the Orchestrator only when research is truly comprehensive."""
    
    def process(self, messages: List[AnyMessage], instruction: str = None, **kwargs) -> Dict[str, Any]:
        """Execute research using dynamic tool selection based on instruction."""
        self.extract_word_count_from_messages(messages)
        if instruction:
            self.extract_word_count_from_instruction(instruction)
        
        outline = self.create_research_outline(instruction, messages)
        
        findings = []
        sections_covered = set()
        
        while self.research_iterations < self.max_iterations:
            context = self.build_research_context(instruction, findings, sections_covered, outline)
            action_plan = self.plan_next_action(messages, context)
            
            if action_plan['action'] == 'respond':
                findings.append({'type': 'insight', 'content': action_plan['input']})
            else:
                result = self.execute_tool(action_plan['action'], action_plan['input'])
                findings.append({
                    'tool': action_plan['action'],
                    'query': action_plan.get('input', ''),
                    'result': result,
                    'reasoning': action_plan['reasoning']
                })
                
                self.update_section_coverage(action_plan, sections_covered)
                
                messages.append(AIMessage(content=f"Research finding: {result[:200]}..."))
            
            self.research_iterations += 1
            
            if self.is_research_complete(findings, sections_covered, outline):
                break
        
        synthesis = self.synthesize_research(findings, outline)
        
        self.research_iterations = 0
        
        return {
            "complete": True,
            "messages": [AIMessage(content=f"Research completed comprehensively. Covered {len(sections_covered)} sections with {len(findings)} findings.\n\nSynthesis: {synthesis[:500]}...")],
            "findings": findings
        }
    
    def extract_word_count_from_messages(self, messages: List[AnyMessage]):
        """Extract target word count from messages."""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                content = msg.content
                match = re.search(r'(\d+)[\s-]?word', content, re.IGNORECASE)
                if match:
                    self.target_word_count = int(match.group(1))
                    self.log(f"Target word count set to: {self.target_word_count}")
    
    def extract_word_count_from_instruction(self, instruction: str):
        """Extract target word count from instruction."""
        if instruction:
            match = re.search(r'(\d+)[\s-]?word', instruction, re.IGNORECASE)
            if match:
                self.target_word_count = int(match.group(1))
                self.log(f"Target word count set to: {self.target_word_count}")
    
    def create_research_outline(self, instruction: str, messages: List[AnyMessage]) -> str:
        """Create research outline using available tools."""
        if self.target_word_count <= 500:
            body_sections = 3
        elif self.target_word_count <= 1000:
            body_sections = 4
        elif self.target_word_count <= 1500:
            body_sections = 5
        else:
            body_sections = 6
        
        outline = None
        for tool in self.tools:
            if 'outline' in tool.name.lower():
                topic = instruction if instruction else "general topic"
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        if "report on:" in msg.content:
                            topic = msg.content.split("report on:")[-1].strip()
                            break
                
                outline = self.execute_tool(tool.name, f"{topic}::{body_sections}")
                
                for tool2 in self.tools:
                    if 'save' in tool2.name.lower() and 'memory' in tool2.name.lower():
                        self.execute_tool(tool2.name, f"report_outline::{outline}")
                        break
                
                self.log(f"Created outline with {body_sections} body sections")
                break
        
        if not outline:
            outline = f"""1. Abstract
2. Introduction
3. Current State and Overview
4. Key Trends and Analysis
5. Challenges and Opportunities
6. Conclusion and Recommendations"""
        
        return outline
    
    def build_research_context(self, instruction: str, findings: List, 
                              sections_covered: set, outline: str) -> str:
        """Build context for planning next research action."""
        sections_in_outline = len([line for line in outline.split('\n') if line.strip()])
        
        context_parts = [
            f"Instruction: {instruction[:100]}..." if instruction else "General research",
            f"Progress: {len(findings)} findings collected",
            f"Sections covered: {len(sections_covered)}/{sections_in_outline}",
            f"Iteration: {self.research_iterations + 1}/{self.max_iterations}"
        ]
        
        # Add info about what sections still need research
        if len(sections_covered) < sections_in_outline:
            context_parts.append("Still need research for some sections")
        
        return " | ".join(context_parts)
    
    def update_section_coverage(self, action_plan: Dict, sections_covered: set):
        """Track which sections have been researched."""
        # Simple heuristic - could be improved with NLP
        query = action_plan.get('input', '').lower()
        reasoning = action_plan.get('reasoning', '').lower()
        
        section_keywords = {
            'introduction': ['overview', 'background', 'introduction', 'basic'],
            'current_state': ['current', 'present', 'state', 'status', 'now'],
            'trends': ['trend', 'development', 'growth', 'change', 'evolution'],
            'challenges': ['challenge', 'problem', 'issue', 'obstacle', 'difficulty'],
            'opportunities': ['opportunity', 'benefit', 'advantage', 'potential'],
            'future': ['future', 'outlook', 'prediction', 'forecast'],
            'conclusion': ['conclusion', 'summary', 'recommendation']
        }
        
        for section, keywords in section_keywords.items():
            if any(keyword in query or keyword in reasoning for keyword in keywords):
                sections_covered.add(section)
    
    def is_research_complete(self, findings: List, sections_covered: set, outline: str) -> bool:
        """Determine if research is comprehensive enough."""
        sections_in_outline = len([line for line in outline.split('\n') if line.strip()])
        
        # Require minimum findings and section coverage
        has_enough_findings = len(findings) >= self.min_findings_required
        has_section_coverage = len(sections_covered) >= max(3, sections_in_outline - 2)
        
        if not has_enough_findings or not has_section_coverage:
            return False
        
        # Check with agent's assessment
        assessment_prompt = f"""Assess if research is complete:
        - Findings collected: {len(findings)}
        - Sections covered: {len(sections_covered)}/{sections_in_outline}
        - Outline sections: {outline}
        
        Is the research comprehensive enough for a high-quality report?
        Consider: depth, breadth, credibility, and completeness.
        
        Answer: COMPLETE or CONTINUE"""
        
        response = self.generate_response([], assessment_prompt)
        return "COMPLETE" in response.upper()
    
    def synthesize_research(self, findings: List, outline: str) -> str:
        """Synthesize all research findings into a comprehensive summary."""
        # First save individual findings
        for i, finding in enumerate(findings):
            if finding.get('tool') == 'web_search':
                # Save search results
                key = f"research_finding_{i}"
                value = f"{finding.get('query', 'Unknown query')}: {finding.get('result', '')[:500]}"
                
                for tool in self.tools:
                    if 'save' in tool.name.lower() and 'memory' in tool.name.lower():
                        self.execute_tool(tool.name, f"{key}::{value}")
                        break
        
        # Create synthesis
        synthesis_prompt = f"""Synthesize all research findings into a comprehensive summary:
        
        Outline structure:
        {outline}
        
        Findings to synthesize:
        {self.format_findings_for_synthesis(findings)}
        
        Create a well-organized synthesis that:
        1. Covers all sections of the outline
        2. Highlights key data and facts
        3. Notes important patterns
        4. Identifies any gaps
        5. Provides a strong foundation for analysis
        
        Organize by outline sections."""
        
        synthesis = self.generate_response([], synthesis_prompt)
        
        # Save synthesis to memory
        for tool in self.tools:
            if 'save' in tool.name.lower() and 'memory' in tool.name.lower():
                self.execute_tool(tool.name, f"research_synthesis::{synthesis}")
                self.execute_tool(tool.name, f"research_summary::{synthesis[:1000]}")
                break
        
        return synthesis
    
    def format_findings_for_synthesis(self, findings: List) -> str:
        """Format findings for synthesis prompt."""
        formatted = []
        for i, finding in enumerate(findings, 1):
            if finding.get('type') == 'insight':
                formatted.append(f"Finding {i}: {finding['content']}")
            else:
                formatted.append(f"Finding {i} (via {finding.get('tool', 'unknown')}): {finding.get('result', '')[:300]}...")
        
        return "\n\n".join(formatted[:10])  # Limit to prevent prompt overflow