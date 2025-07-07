from typing import List, Dict, Any, Optional
from langchain_core.messages import AnyMessage, AIMessage
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.base_agent import BaseAgent
from src.tools import ANALYSIS_TOOLS

class AnalysisAgent(BaseAgent):
    """Analysis agent for the orchestrator team using dynamic tool selection."""
    
    def __init__(self):
        super().__init__("Analysis Agent (Orchestrator Team)", ANALYSIS_TOOLS)
        self.analysis_iterations = 0
        self.max_iterations = 6
        self.min_insights_required = 4
    
    def get_system_prompt(self) -> str:
        return """You are an Analysis Agent working under an Orchestrator's guidance.
        
        Your responsibilities:
        1. Read and synthesize ALL research findings from shared memory
        2. Identify patterns, trends, and key insights using analytical tools
        3. Perform calculations and comparisons when needed
        4. Check for factual consistency across sources
        5. Draw meaningful, actionable conclusions
        6. Prepare structured analysis for the Writer agent
        
        Use your tools strategically:
        - Use read_from_memory to access all research data
        - Use calculator for quantitative analysis
        - Use extract_facts to identify key points
        - Use check_factual_consistency to validate claims
        - Use save_to_memory to store insights and analysis
        
        Be thorough and critical - provide deep insights, not surface observations.
        Your analysis should enable the creation of a high-quality report."""
    
    def process(self, messages: List[AnyMessage], instruction: str = None, **kwargs) -> Dict[str, Any]:
        """Analyze research findings using dynamic tool selection."""
        # Initialize analysis
        analysis_results = []
        insights_generated = []
        
        # Main analysis loop
        while self.analysis_iterations < self.max_iterations:
            # Build context for planning
            context = self.build_analysis_context(instruction, analysis_results, insights_generated)
            
            # Plan next analytical action
            action_plan = self.plan_next_action(messages, context)
            
            # Execute planned action
            if action_plan['action'] == 'respond':
                # Direct analytical insight
                analysis_results.append({
                    'type': 'insight',
                    'content': action_plan['input']
                })
            else:
                # Execute analytical tool
                result = self.execute_tool(action_plan['action'], action_plan['input'])
                analysis_results.append({
                    'tool': action_plan['action'],
                    'input': action_plan.get('input', ''),
                    'result': result,
                    'reasoning': action_plan['reasoning']
                })
                
                # Update messages for context
                messages.append(AIMessage(content=f"Analysis result: {result[:200]}..."))
            
            self.analysis_iterations += 1
            
            # Check if analysis is sufficient
            if self.is_analysis_complete(analysis_results):
                break
        
        # Generate final insights and recommendations
        final_insights = self.generate_comprehensive_insights(analysis_results)
        consistency_report = self.generate_consistency_report(analysis_results)
        
        # Save everything to memory
        self.save_analysis_results(analysis_results, final_insights, consistency_report)
        
        # Reset for next time
        self.analysis_iterations = 0
        
        return {
            "complete": True,
            "messages": [AIMessage(content=f"Analysis completed with {len(analysis_results)} analytical operations.\n\nKey Insights:\n{final_insights[:500]}...")],
            "analysis": analysis_results,
            "insights": final_insights
        }
    
    def build_analysis_context(self, instruction: str, results: List, insights: List) -> str:
        """Build context for planning next analytical action."""
        # Check what data is available
        available_data = self.check_available_research_data()
        
        context_parts = [
            f"Instruction: {instruction[:100]}..." if instruction else "Analyze research findings",
            f"Progress: {len(results)} analyses performed",
            f"Insights found: {len(insights)}",
            f"Available data: {available_data}",
            f"Iteration: {self.analysis_iterations + 1}/{self.max_iterations}"
        ]
        
        return " | ".join(context_parts)
    
    def check_available_research_data(self) -> str:
        """Check what research data is available in memory."""
        available = []
        
        # Check for common research keys
        research_keys = ['research_synthesis', 'research_summary', 'report_outline']
        
        for tool in self.tools:
            if 'list' in tool.name.lower() and 'memory' in tool.name.lower():
                memory_keys = self.execute_tool(tool.name, '')
                for key in research_keys:
                    if key in memory_keys:
                        available.append(key)
                break
        
        # Also check for numbered findings
        if 'memory' in str(self.tools):
            for tool in self.tools:
                if 'list' in tool.name.lower():
                    keys = self.execute_tool(tool.name, '')
                    finding_count = len([k for k in keys.split(',') if 'research_finding' in k])
                    if finding_count > 0:
                        available.append(f"{finding_count} research findings")
                    break
        
        return ", ".join(available) if available else "No research data found"
    
    def is_analysis_complete(self, results: List) -> bool:
        """Determine if analysis is comprehensive enough."""
        # Count different types of analysis performed
        tools_used = set()
        insights_count = 0
        
        for result in results:
            if result.get('type') == 'insight':
                insights_count += 1
            elif result.get('tool'):
                tools_used.add(result['tool'])
        
        # Require diverse analysis
        has_enough_results = len(results) >= self.min_insights_required
        has_tool_diversity = len(tools_used) >= 2
        
        if not has_enough_results or not has_tool_diversity:
            return False
        
        # Self-assessment
        assessment_prompt = f"""Assess if analysis is complete:
        - Analyses performed: {len(results)}
        - Different tools used: {len(tools_used)}
        - Direct insights: {insights_count}
        
        Have we:
        1. Thoroughly analyzed all research data?
        2. Identified key patterns and trends?
        3. Checked data consistency?
        4. Generated actionable insights?
        5. Prepared sufficient foundation for report writing?
        
        Answer: COMPLETE or CONTINUE"""
        
        response = self.generate_response([], assessment_prompt)
        return "COMPLETE" in response.upper()
    
    def generate_comprehensive_insights(self, analysis_results: List) -> str:
        """Generate comprehensive insights from all analyses."""
        insights_prompt = f"""Based on all analytical work performed:
        
        {self.format_analysis_results(analysis_results)}
        
        Generate 4-6 KEY INSIGHTS that are:
        1. Specific and data-driven
        2. Actionable and practical
        3. Well-supported by the analysis
        4. Relevant to the report objectives
        5. Forward-looking where appropriate
        
        Format each insight with:
        - Clear statement
        - Supporting evidence
        - Implications
        
        These insights will drive the final report."""
        
        return self.generate_response([], insights_prompt)
    
    def generate_consistency_report(self, analysis_results: List) -> str:
        """Generate report on data consistency and reliability."""
        # Extract consistency checks from results
        consistency_checks = []
        for result in analysis_results:
            if result.get('tool') == 'check_factual_consistency':
                consistency_checks.append(result)
        
        if not consistency_checks:
            return "No formal consistency checks performed."
        
        report_prompt = f"""Based on consistency checks performed:
        
        {self.format_consistency_checks(consistency_checks)}
        
        Provide:
        1. Overall assessment of data reliability
        2. Any contradictions or concerns found
        3. Confidence level in the findings
        4. Recommendations for the writer
        
        Be honest about any limitations or uncertainties."""
        
        return self.generate_response([], report_prompt)
    
    def save_analysis_results(self, results: List, insights: str, consistency: str):
        """Save all analysis results to memory."""
        # Find save_to_memory tool
        save_tool = None
        for tool in self.tools:
            if 'save' in tool.name.lower() and 'memory' in tool.name.lower():
                save_tool = tool
                break
        
        if save_tool:
            # Save main components
            self.execute_tool(save_tool.name, f"analysis_complete::true")
            self.execute_tool(save_tool.name, f"analysis_insights::{insights}")
            self.execute_tool(save_tool.name, f"key_insights::{insights[:1000]}")
            self.execute_tool(save_tool.name, f"consistency_report::{consistency}")
            
            # Save detailed results
            analysis_summary = self.create_analysis_summary(results)
            self.execute_tool(save_tool.name, f"analysis_results::{analysis_summary}")
    
    def format_analysis_results(self, results: List) -> str:
        """Format analysis results for prompt."""
        formatted = []
        for i, result in enumerate(results[:8], 1):  # Limit to prevent overflow
            if result.get('type') == 'insight':
                formatted.append(f"Insight {i}: {result['content']}")
            else:
                formatted.append(f"Analysis {i} (using {result.get('tool', 'unknown')}): {result.get('result', '')[:200]}...")
        
        return "\n\n".join(formatted)
    
    def format_consistency_checks(self, checks: List) -> str:
        """Format consistency checks for prompt."""
        formatted = []
        for i, check in enumerate(checks[:5], 1):
            formatted.append(f"Check {i}: {check.get('input', 'Unknown comparison')}\nResult: {check.get('result', 'Unknown result')}")
        
        return "\n\n".join(formatted)
    
    def create_analysis_summary(self, results: List) -> str:
        """Create a summary of all analysis performed."""
        summary_parts = [
            f"Total analyses: {len(results)}",
            f"Tools used: {', '.join(set(r.get('tool', '') for r in results if r.get('tool')))}",
            f"Key findings: {len([r for r in results if r.get('type') == 'insight'])}"
        ]
        
        # Add brief overview of findings
        key_points = []
        for result in results[:5]:
            if result.get('type') == 'insight':
                key_points.append(result['content'][:100] + "...")
            elif result.get('result'):
                key_points.append(result['result'][:100] + "...")
        
        if key_points:
            summary_parts.append("Key points:\n" + "\n".join(f"- {point}" for point in key_points))
        
        return "\n\n".join(summary_parts)