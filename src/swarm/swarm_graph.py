from typing import TypedDict, List, Annotated, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.swarm.research_agent import SwarmResearchAgent
from src.swarm.analysis_agent import SwarmAnalysisAgent
from src.swarm.writer_agent import SwarmWriterAgent
from src.logging_utils import AgentLogger, save_report, log_agent_communication

# Define state structure
class SwarmState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    current_agent: str
    task_complete: bool
    from_agent: Optional[str]
    instruction_for_next: Optional[str]
    report: Optional[str]
    iteration_count: int
    logger: AgentLogger  # Add logger to state</    

# Initialize agents
research_agent = SwarmResearchAgent()
analysis_agent = SwarmAnalysisAgent()
writer_agent = SwarmWriterAgent()

def research_node(state: SwarmState) -> SwarmState:
    """Research agent node."""
    research_agent.set_logger(state["logger"])
    
    from_agent = state.get("from_agent")
    if from_agent:
        research_agent.log(f"Received request from {from_agent}")
    else:
        research_agent.log("Starting initial research")
    
    result = research_agent.process(
        state["messages"],
        from_agent=from_agent,
        instruction=state.get("instruction_for_next")
    )
    
    new_state = state.copy()
    new_state["messages"].extend(result["messages"])
    new_state["from_agent"] = "research"
    
    if result.get("next_agent"):
        new_state["current_agent"] = result["next_agent"]
        new_state["instruction_for_next"] = result.get("instruction_for_next")
        
        # Log peer communication
        log_agent_communication(
            state["logger"],
            "Research Agent",
            result["next_agent"],
            "Peer-to-peer handoff",
            result.get("instruction_for_next", "")
        )
    
    return new_state

def analysis_node(state: SwarmState) -> SwarmState:
    """Analysis agent node."""
    analysis_agent.set_logger(state["logger"])
    
    from_agent = state.get("from_agent")
    analysis_agent.log(f"Received request from {from_agent}")
    
    result = analysis_agent.process(
        state["messages"],
        from_agent=from_agent,
        instruction=state.get("instruction_for_next")
    )
    
    new_state = state.copy()
    new_state["messages"].extend(result["messages"])
    new_state["from_agent"] = "analysis"
    
    if result.get("next_agent"):
        new_state["current_agent"] = result["next_agent"]
        new_state["instruction_for_next"] = result.get("instruction_for_next")
        
        # Log peer communication
        log_agent_communication(
            state["logger"],
            "Analysis Agent",
            result["next_agent"],
            "Peer-to-peer handoff",
            result.get("instruction_for_next", "")
        )
    
    return new_state

def writer_node(state: SwarmState) -> SwarmState:
    """Writer agent node - can declare task complete."""
    writer_agent.set_logger(state["logger"])
    
    from_agent = state.get("from_agent")
    writer_agent.log(f"Received request from {from_agent}")
    
    result = writer_agent.process(
        state["messages"],
        from_agent=from_agent,
        instruction=state.get("instruction_for_next")
    )
    
    new_state = state.copy()
    new_state["messages"].extend(result["messages"])
    new_state["from_agent"] = "writer"
    
    if result.get("complete") and result.get("next_agent") is None:
        # Task is complete
        new_state["task_complete"] = True
        new_state["report"] = result.get("report", "")
        writer_agent.log("Task declared complete - report finalized")
    elif result.get("next_agent"):
        new_state["current_agent"] = result["next_agent"]
        new_state["instruction_for_next"] = result.get("instruction_for_next")
        
        # Log peer communication
        log_agent_communication(
            state["logger"],
            "Writer Agent",
            result["next_agent"],
            "Requesting additional work",
            result.get("instruction_for_next", "")
        )
    
    return new_state

def build_swarm_graph():
    """Build the swarm team graph with peer-to-peer communication."""
    workflow = StateGraph(SwarmState)
    
    # Add nodes
    workflow.add_node("research", research_node)
    workflow.add_node("analysis", analysis_node) 
    workflow.add_node("writer", writer_node)
    
    # Entry point - always start with research
    workflow.add_edge(START, "research")
    
    # Dynamic routing based on agent decisions
    workflow.add_conditional_edges(
        "research",
        lambda state: state.get("current_agent", "analysis"),
        {
            "research": "research",
            "analysis": "analysis", 
            "writer": "writer"
        }
    )
    
    workflow.add_conditional_edges(
        "analysis",
        lambda state: state.get("current_agent", "writer"),
        {
            "research": "research",
            "analysis": "analysis",
            "writer": "writer"
        }
    )
    
    workflow.add_conditional_edges(
        "writer",
        lambda state: END if state.get("task_complete") else state.get("current_agent", "analysis"),
        {
            "research": "research",
            "analysis": "analysis",
            "writer": "writer",
            END: END
        }
    )
    
    return workflow.compile()

def run_swarm_team(task: str, word_count: int = 500) -> str:
    """Run the swarm team on a given task."""
    # Clear shared memory at start
    import os
    if os.path.exists('shared_memory.json'):
        os.remove('shared_memory.json')
    
    # Initialize logger
    logger = AgentLogger("swarm")
    logger.log("System", f"Starting swarm team for task: {task}")
    logger.log("System", f"Target word count: {word_count}")
    logger.log_separator("Task Execution - Peer-to-Peer Communication")
    
    graph = build_swarm_graph()
    
    initial_state = {
        "messages": [HumanMessage(content=f"Create a {word_count}-word report on: {task}")],
        "current_agent": "research",
        "task_complete": False,
        "from_agent": None,
        "instruction_for_next": None,
        "report": None,
        "iteration_count": 0,
        "logger": logger
    }
    
    # Run the graph
    result = graph.invoke(initial_state)
    
    # Get report from state or memory
    report = result.get("report")
    if not report:
        # Try to get from memory as fallback
        try:
            from tools import read_from_memory
            report = read_from_memory.invoke("final_report")
            if "No data found" not in report:
                logger.log("System", "Retrieved report from shared memory")
            else:
                report = "Report generation did not complete successfully."
                logger.log("System", "Report generation failed - no report found")
        except Exception as e:
            logger.log("System", f"Error retrieving report: {str(e)}")
            report = "Report generation did not complete successfully."
    
    # Save the report if successful
    if report and "Report generation did not complete" not in report:
        logger.log_separator("Final Report")
        logger.log("System", "Report generation completed successfully")
        
        report_file = save_report(
            report,
            "swarm",
            task,
            logger.get_log_number()
        )
        logger.log("System", f"Report saved to: {report_file}")
    
    logger.log_separator("End of Execution")
    
    return report