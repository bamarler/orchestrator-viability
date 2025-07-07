from typing import List, Dict, Any, Optional
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.base_agent import BaseAgent
from src.tools import ANALYSIS_TOOLS

class SwarmAnalysisAgent(BaseAgent):
    """Analysis agent for the swarm team that can communicate directly with other agents."""
    
    def __init__(self):
        super().__init__("Analysis Agent (Swarm Team)", ANALYSIS_TOOLS)
        self.analysis_depth = "standard"
        self.analysis_iterations = 0
        self.max_iterations = 4
    
    def get_system_prompt(self) -> str:
        return """You are an Analysis Agent in a decentralized swarm team.
        
        Your responsibilities:
        1. Analyze research findings using available analytical tools
        2. Identify patterns, trends, and insights
        3. Perform calculations and statistical analysis
        4. Check factual consistency and validate data
        5. Generate actionable insights
        6. Decide if more research is needed or proceed to writing
        
        You work autonomously and can:
        - Use memory tools to access and store data
        - Use calculation tools for quantitative analysis
        - Use consistency checking tools for validation
        - Use comparison tools for pattern recognition
        - Request more research if gaps are found
        - Pass completed analysis to the Writer agent
        
        Your analysis should be data-driven and actionable."""
    
    def process(self, messages: List[AnyMessage], from_agent: Optional[str] = None, 
                instruction: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Process analysis task using dynamic tool selection."""
        
        # Determine analysis context
        if from_agent == "writer" and "deeper" in str(instruction).lower():
            self.analysis_depth = "deep"
        
        # Initialize analysis
        analysis_results = []
        insights = []
        
        # Main analysis loop
        while self.analysis_iterations < self.max_iterations:
            # Determine context for planning
            context = self.build_analysis_context(from_agent, instruction, analysis_results)
            
            # Plan next analytical action
            action_plan = self.plan_next_action(messages, context)
            
            # Execute planned action
            if action_plan['action'] == 'respond':
                # Direct response
                analysis_results.append({
                    'type': 'insight',
                    'content': action_plan['input']
                })
            else:
                # Execute analytical tool
                result = self.execute_tool(action_plan['action'], action_plan['input'])
                analysis_results.append({
                    'tool': action_plan['action'],
                    'result': result,
                    'reasoning': action_plan['reasoning']
                })
                
                # Update messages with result
                messages.append(AIMessage(content=f"Analysis using {action_plan['action']}: {result[:200]}..."))
            
            self.analysis_iterations += 1
            
            # Check if analysis is sufficient
            if self.is_analysis_complete(analysis_results, self.analysis_depth):
                break
        
        # Generate final insights
        final_insights = self.synthesize_insights(analysis_results)
        
        # Determine next steps
        return self.determine_next_steps(final_insights, from_agent)
    
    def build_analysis_context(self, from_agent: Optional[str], 
                              instruction: Optional[str], 
                              current_results: List) -> str:
        """Build context string for action planning."""
        context_parts = [
            f"Analysis depth: {self.analysis_depth}",
            f"Iteration: {self.analysis_iterations + 1}/{self.max_iterations}",
            f"Results so far: {len(current_results)} findings"
        ]
        
        if from_agent:
            context_parts.append(f"Request from: {from_agent}")
        
        if instruction:
            context_parts.append(f"Specific focus: {instruction[:100]}...")
        
        # Check what data is available
        data_availability = self.check_data_availability()
        context_parts.append(f"Available data: {data_availability}")
        
        return " | ".join(context_parts)
    
    def check_data_availability(self) -> str:
        """Check what data is available in memory."""
        available_data = []
        
        # Try to check memory for common keys
        for tool in self.tools:
            if 'memory' in tool.name.lower() and 'read' in tool.name.lower():
                # Check for research data
                research_check = self.execute_tool(tool.name, 'research_synthesis')
                if "No data found" not in research_check:
                    available_data.append("research findings")
                break
        
        return ", ".join(available_data) if available_data else "No data in memory"
    
    def is_analysis_complete(self, results: List, depth: str) -> bool:
        """Determine if analysis is complete."""
        min_results = 2 if depth == "standard" else 4
        
        if len(results) < min_results:
            return False
        
        # Check if we have diverse analysis types
        tool_types = set()
        for result in results:
            if 'tool' in result:
                tool_types.add(result['tool'])
        
        return len(tool_types) >= 2 or self.analysis_iterations >= self.max_iterations
    
    def synthesize_insights(self, analysis_results: List) -> Dict[str, Any]:
        """Synthesize all analysis results into key insights."""
        synthesis_prompt = f"""Synthesize these analysis results into key insights:
        
        {self.format_analysis_results(analysis_results)}
        
        Generate:
        1. 3-5 key insights that are specific and actionable
        2. Main patterns or trends identified
        3. Any data inconsistencies or concerns
        4. Recommendations based on the analysis
        
        Format each insight clearly and support with data."""
        
        insights_text = self.generate_response([], synthesis_prompt)
        
        # Save insights to memory if possible
        for tool in self.tools:
            if 'memory' in tool.name.lower() and 'save' in tool.name.lower():
                self.execute_tool(tool.name, f"analysis_insights::{insights_text}")
                self.execute_tool(tool.name, f"analysis_complete::true")
                break
        
        return {
            'insights': insights_text,
            'analysis_count': len(analysis_results),
            'depth': self.analysis_depth
        }
    
    def format_analysis_results(self, results: List) -> str:
        """Format analysis results for synthesis."""
        formatted = []
        for i, result in enumerate(results, 1):
            if result.get('type') == 'insight':
                formatted.append(f"Insight {i}: {result['content']}")
            else:
                formatted.append(f"Analysis {i} (using {result.get('tool', 'unknown')}):\n{result.get('result', '')[:300]}...")
        return "\n\n".join(formatted)
    
    def determine_next_steps(self, insights: Dict[str, Any], 
                           from_agent: Optional[str]) -> Dict[str, Any]:
        """Determine next steps based on analysis results."""
        # Check if we need more research
        needs_more_research = self.assess_research_needs(insights['insights'])
        
        # Reset for next time
        self.analysis_iterations = 0
        self.analysis_depth = "standard"
        
        if needs_more_research and from_agent != "research":
            return {
                "next_agent": "research",
                "complete": False,
                "instruction_for_next": needs_more_research,
                "messages": [AIMessage(content=f"Analysis reveals gaps. Requesting additional research:\n{needs_more_research}")]
            }
        else:
            # Prepare for writer
            writer_instruction = self.prepare_writer_briefing(insights)
            
            return {
                "next_agent": "writer",
                "complete": True,
                "instruction_for_next": writer_instruction,
                "messages": [AIMessage(content=f"Analysis complete. Key insights:\n{insights['insights'][:500]}...\n\nHanding off to Writer agent.")]
            }
    
    def assess_research_needs(self, insights: str) -> Optional[str]:
        """Assess if more research is needed - be more demanding."""
        assessment_prompt = f"""Based on these analysis insights:
        {insights[:500]}...
        
        Determine if additional research is needed for a COMPREHENSIVE report.
        Check for:
        - Missing data for Abstract (summary overview)
        - Insufficient context for Introduction
        - Lack of depth in any of the body sections (need at least 3)
        - Missing support for Conclusions/Recommendations
        - Lack of recent statistics or numbers
        - Unverified or vague claims
        - Missing perspectives or counterarguments
        
        Be demanding - if ANY section lacks depth, request more research.
        
        If research is needed, specify EXACTLY what is missing.
        If truly complete, respond with 'NONE'.
        
        Response format:
        NEED_RESEARCH: [YES/NO]
        SPECIFICS: [detailed list of what research is needed, or 'NONE']"""
        
        response = self.generate_response([], assessment_prompt)
        
        if "NEED_RESEARCH: YES" in response:
            for line in response.split('\n'):
                if line.startswith('SPECIFICS:'):
                    specifics = line.replace('SPECIFICS:', '').strip()
                    if specifics and specifics != 'NONE':
                        return f"Need more research on: {specifics}"
        
        return None
    
    def prepare_writer_briefing(self, insights: Dict[str, Any]) -> str:
        """Prepare briefing for the writer agent."""
        briefing_prompt = f"""Create a clear briefing for the Writer agent based on:
        
        Analysis insights: {insights['insights'][:600]}...
        Analysis depth: {insights['depth']}
        Number of analyses: {insights['analysis_count']}
        
        The briefing should:
        1. Highlight the most important findings
        2. Suggest report structure
        3. Identify key messages
        4. Note any caveats or limitations
        
        Keep it concise and actionable."""
        
        return self.generate_response([], briefing_prompt)