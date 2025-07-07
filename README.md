# Multi-Agent Orchestration Benchmark

The purpose of this project is to compare two multi-agent architectures for collaborative report generation:
1. **Orchestrator Team**: Centralized coordination with an orchestrator managing three specialized agents
2. **Swarm Team**: Decentralized peer-to-peer communication between three autonomous agents

## Agent Roles

### Research Agent
- Searches for relevant information
- Extracts key facts and data
- Saves findings to shared memory
- Verifies information across sources

### Analysis Agent
- Synthesizes research findings
- Identifies patterns and trends
- Performs calculations and comparisons
- Generates actionable insights

### Writer Agent
- Creates structured reports
- Maintains coherent narrative
- Ensures proper flow and style
- Determines when task is complete

## Key Differences

### Orchestrator Team
- **Communication**: All through central orchestrator
- **Decision Making**: Orchestrator plans and assigns tasks
- **Advantages**: Clear control flow, consistent coordination
- **Entry/Exit**: Orchestrator manages entire workflow

### Swarm Team
- **Communication**: Direct peer-to-peer between agents
- **Decision Making**: Agents autonomously decide next steps
- **Advantages**: Parallel processing, emergent solutions
- **Entry/Exit**: Research agent entry, Writer agent determines completion

## Project Structure

```
src/
├── analysis/
│   ├── swarm_log_analyzer.py
│   ├── orchestrator_log_analyzer.py
├── orchestrator_team/         
│   ├── __init__.py
│   ├── orchestrator_agent.py  
│   ├── research_agent.py      
│   ├── analysis_agent.py      
│   ├── writer_agent.py        
│   └── orchestrator_graph.py  
├── swarm_team/               
│   ├── __init__.py
│   ├── research_agent.py      
│   ├── analysis_agent.py      
│   ├── writer_agent.py        
│   └── swarm_graph.py     
├── __init__.py                                     
├── tools.py                   
├── base_agent.py              
├── logging_utils.py      
├── run_orchestrator.py        # Easy runner for orchestrator team 
└── run_swarm.py               # Easy runner for swarm team

Output directories (created automatically):
├── logs/                      # Detailed execution logs
│   ├── orchestrator_log_1.txt 
│   └── swarm_log_1.txt        
└── reports/                   # Generated reports
    ├── orchestrator_report_1_*.md
    └── swarm_report_1_*.md
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the src directory (or parent directory) with the following content:
```
GEMINI_API_KEY=your_api_key_here
```

## Running the System

To run either team, you can run the two scripts `run_orchestrator.py` and `run_swarm.py` in the `src` directory. You can modify how long of a report each team generates by modifying the `report_length` parameter and what the team searches for by modifying the `task` parameters.

## Logging and Output

### Logs
Each run generates a detailed log file with timestamped entries:
- **Orchestrator logs**: Show hierarchical communication flow
  - Orchestrator decisions and task assignments
  - Agent responses and completions
  - Tool executions with inputs/outputs
  
- **Swarm logs**: Show peer-to-peer interactions
  - Direct agent-to-agent communications
  - Autonomous decision making
  - Collaborative handoffs

### Reports
Final reports are saved as Markdown files with:
- Task description
- Generation timestamp
- Link to corresponding log file
- Full report content

### Log Entry Format
```
[HH:MM:SS] Agent Name: Message | Metadata: {...}
```

## Benchmark Metrics

The benchmark evaluates both approaches on:

**Efficiency Metrics:**
- Time to completion
- Token usage
- API calls
- Inter-agent communications

**Quality Metrics:**
- Coherence score
- Coverage score
- Factual accuracy
- Structure score

**Coordination Metrics:**
- Successful handoffs
- Error recovery
- Adaptive replanning

## Test Scenarios

1. **Simple**: Basic factual report (500 words)
2. **Medium**: Analysis with pros/cons (750 words)
3. **Complex**: Strategic recommendations (1000 words)

## Expected Insights

- **Orchestrator**: Better for tasks requiring strict coordination and consistency
- **Swarm**: Better for tasks benefiting from parallel exploration and emergent solutions

## Requirements

- Python 3.8+
- Google Gemini API key (Free from <https://aistudio.google.com/apikey>)