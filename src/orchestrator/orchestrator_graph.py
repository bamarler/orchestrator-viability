from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.orchestrator.orchestrator_agent import OrchestratorAgent
from src.orchestrator.research_agent import ResearchAgent
from src.orchestrator.analysis_agent import AnalysisAgent
from src.orchestrator.writer_agent import WriterAgent
from src.logging_utils import AgentLogger, save_report, log_agent_communication

class OrchestratorState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    current_agent: str
    task_complete: bool
    current_instruction: str
    report: str
    iteration_count: int
    max_iterations: int
    logger: AgentLogger

orchestrator = OrchestratorAgent()
research_agent = ResearchAgent()
analysis_agent = AnalysisAgent()
writer_agent = WriterAgent()

def orchestrator_node(state: OrchestratorState) -> OrchestratorState:
    """Orchestrator decides next steps."""
    orchestrator.set_logger(state["logger"])
    orchestrator.log("Processing task and determining next agent")
    
    result = orchestrator.process(state["messages"])
    
    new_state = state.copy()
    new_state["messages"].extend(result["messages"])
    new_state["iteration_count"] += 1
    
    if result["next_agent"]:
        new_state["current_agent"] = result["next_agent"]
        new_state["current_instruction"] = result["instruction"]
        
        log_agent_communication(
            state["logger"],
            "Orchestrator",
            result["next_agent"],
            f"Delegating task (iteration {new_state['iteration_count']})",
            result["instruction"]
        )
    else:
        new_state["task_complete"] = True
        orchestrator.log("Task marked as complete")
        
    return new_state

def research_node(state: OrchestratorState) -> OrchestratorState:
    """Research agent performs research."""
    research_agent.set_logger(state["logger"])
    research_agent.log(f"Starting research with instruction: {state['current_instruction'][:100]}...")
    
    result = research_agent.process(
        state["messages"], 
        instruction=state["current_instruction"]
    )
    
    new_state = state.copy()
    new_state["messages"].extend(result["messages"])
    new_state["iteration_count"] += 1
    
    if result["complete"]:
        orchestrator.update_task_status("research", True)
        research_agent.log("Research phase completed")
    
    log_agent_communication(
        state["logger"],
        "Research Agent",
        "Orchestrator",
        "Research complete, returning results",
        f"Found {len(result.get('findings', []))} findings"
    )
    
    new_state["current_agent"] = "orchestrator"
    
    return new_state

def analysis_node(state: OrchestratorState) -> OrchestratorState:
    """Analysis agent analyzes findings."""
    analysis_agent.set_logger(state["logger"])
    analysis_agent.log(f"Starting analysis with instruction: {state['current_instruction'][:100]}...")
    
    result = analysis_agent.process(
        state["messages"],
        instruction=state["current_instruction"]
    )
    
    new_state = state.copy()
    new_state["messages"].extend(result["messages"])
    new_state["iteration_count"] += 1
    
    if result["complete"]:
        orchestrator.update_task_status("analysis", True)
        analysis_agent.log("Analysis phase completed")
    
    log_agent_communication(
        state["logger"],
        "Analysis Agent",
        "Orchestrator",
        "Analysis complete, returning insights",
        f"Generated {len(result.get('analysis', []))} analyses"
    )
    
    new_state["current_agent"] = "orchestrator"
    
    return new_state

def writer_node(state: OrchestratorState) -> OrchestratorState:
    """Writer agent creates final report."""
    writer_agent.set_logger(state["logger"])
    writer_agent.log(f"Starting report writing with instruction: {state['current_instruction'][:100]}...")
    
    result = writer_agent.process(
        state["messages"],
        instruction=state["current_instruction"]
    )
    
    new_state = state.copy()
    new_state["messages"].extend(result["messages"])
    new_state["iteration_count"] += 1
    
    if result["complete"]:
        new_state["report"] = result["report"]
        orchestrator.update_task_status("writer", True)
        writer_agent.log(f"Report completed. Word count: {result.get('word_count', 'Unknown')}")
    
    log_agent_communication(
        state["logger"],
        "Writer Agent",
        "Orchestrator",
        "Writing complete, report ready",
        f"Word count: {result.get('word_count', 'Unknown')}"
    )
    
    new_state["current_agent"] = "orchestrator"
    
    return new_state

def route_next_agent(state: OrchestratorState) -> str:
    """Route to the next agent based on current state."""
    if state["task_complete"]:
        return END
    
    if state["iteration_count"] >= state["max_iterations"]:
        state["logger"].log("System", f"Maximum iterations ({state['max_iterations']}) reached.")
        return END
    
    agent = state["current_agent"]
    if agent == "orchestrator":
        return "orchestrator"
    elif agent == "research":
        return "research"
    elif agent == "analysis":
        return "analysis"
    elif agent == "writer":
        return "writer"
    else:
        return "orchestrator"

def build_orchestrator_graph():
    """Build the orchestrator team graph."""
    workflow = StateGraph(OrchestratorState)
    
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("research", research_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("writer", writer_node)
    
    workflow.add_edge(START, "orchestrator")
    
    workflow.add_conditional_edges(
        "orchestrator",
        route_next_agent,
        {
            "orchestrator": "orchestrator",
            "research": "research",
            "analysis": "analysis",
            "writer": "writer",
            END: END
        }
    )
    
    workflow.add_edge("research", "orchestrator")
    workflow.add_edge("analysis", "orchestrator")
    workflow.add_edge("writer", "orchestrator")
    
    return workflow.compile()

def run_orchestrator_team(task: str, word_count: int = 500) -> str:
    """Run the orchestrator team on a given task."""
    if os.path.exists('shared_memory.json'):
        os.remove('shared_memory.json')
    
    logger = AgentLogger("orchestrator")
    logger.log("System", f"Starting orchestrator team for task: {task}")
    logger.log("System", f"Target word count: {word_count}")
    logger.log_separator("Task Execution - Centralized Orchestration")
    
    orchestrator.set_logger(logger)
    
    graph = build_orchestrator_graph()
    
    initial_state = {
        "messages": [HumanMessage(content=f"Create a {word_count}-word report on: {task}")],
        "current_agent": "orchestrator",
        "task_complete": False,
        "current_instruction": "",
        "report": "",
        "iteration_count": 0,
        "max_iterations": 25,
        "logger": logger
    }
    
    result = graph.invoke(initial_state)
    
    report = result.get("report")
    if not report:
        try:
            from src.tools import read_from_memory
            report = read_from_memory.invoke("final_report")
            if "No data found" not in report:
                logger.log("System", "Retrieved report from shared memory")
            else:
                report = "Report generation did not complete successfully."
                logger.log("System", "Report generation failed - no report found")
        except Exception as e:
            logger.log("System", f"Error retrieving report: {str(e)}")
            report = "Report generation did not complete successfully."
    
    if report and "Report generation did not complete" not in report:
        logger.log_separator("Final Report")
        logger.log("System", "Report generation completed successfully")
        logger.log("System", f"Total iterations: {result['iteration_count']}")
        
        report_file = save_report(
            report,
            "orchestrator",
            task,
            logger.get_log_number()
        )
        logger.log("System", f"Report saved to: {report_file}")
    else:
        logger.log("System", "Report generation incomplete - check logs for issues")
    
    logger.log_separator("End of Execution")
    
    return report