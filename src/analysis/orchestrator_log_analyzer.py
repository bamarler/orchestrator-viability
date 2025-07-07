#!/usr/bin/env python3
"""
Orchestrator Team Performance Analyzer
Analyzes orchestrator team logs and creates visualizations of team performance metrics.
"""

import re
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from datetime import datetime
import numpy as np
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class OrchestratorLogAnalyzer:
    def __init__(self, log_content):
        self.log_content = log_content
        self.agents = set()
        self.tool_calls = defaultdict(list)
        self.api_calls = defaultdict(int)
        self.execution_flow = []
        self.task_name = ""
        self.start_time = None
        self.end_time = None
        
    def parse_log(self):
        """Parse the log file and extract metrics."""
        lines = self.log_content.strip().split('\n')
        current_agent = None
        
        for i, line in enumerate(lines):
            # Extract task name
            if "Starting orchestrator team for task:" in line:
                match = re.search(r'task: (.+?)$', line)
                if match:
                    self.task_name = match.group(1)
            
            # Extract timestamp
            time_match = re.match(r'\[(\d{2}:\d{2}:\d{2})\]', line)
            if time_match:
                timestamp = time_match.group(1)
                if not self.start_time:
                    self.start_time = timestamp
                self.end_time = timestamp
            
            # Extract agent name - handle both Orchestrator and other agents
            # Pattern 1: "Orchestrator:" 
            if line.startswith('[') and 'Orchestrator:' in line:
                agent = 'Orchestrator'
                self.agents.add(agent)
                current_agent = agent
                
                # Track execution flow
                if not self.execution_flow or self.execution_flow[-1] != agent:
                    self.execution_flow.append(agent)
                
                # Check for API calls (response generation)
                if "Generating response" in line:
                    self.api_calls[agent] += 1
            
            # Pattern 2: "Agent Name (Orchestrator Team):"
            agent_match = re.search(r'(\w+ Agent) \(Orchestrator Team\):', line)
            if agent_match:
                agent = agent_match.group(1)
                self.agents.add(agent)
                current_agent = agent
                
                # Track execution flow
                if not self.execution_flow or self.execution_flow[-1] != agent:
                    self.execution_flow.append(agent)
                
                # Check for API calls (response generation)
                if "Generating response" in line:
                    self.api_calls[agent] += 1
            
            # Check for planned actions (tool calls) - can happen on any line after agent is set
            if current_agent and "Planned action:" in line:
                # Look for the action in the metadata
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    action_match = re.search(r'"action": "(\w+)"', next_line)
                    if action_match:
                        action = action_match.group(1)
                        self.tool_calls[current_agent].append(action)
    
    def create_visualization(self, output_path):
        """Create comprehensive visualization of team performance."""
        # Set up the figure with subplots
        fig = plt.figure(figsize=(16, 10))
        
        # Create title with both analysis type and task
        main_title = f'Orchestrator Log Analysis: {self.task_name}'
        fig.suptitle(main_title, fontsize=20, fontweight='bold', y=0.98)
        
        # Add runtime info with better styling
        runtime_text = f"Runtime: {self.start_time} - {self.end_time}"
        fig.text(0.5, 0.93, runtime_text, ha='center', fontsize=11, style='italic', color='#4B0082')
        
        # Create grid layout with better spacing
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1.5], 
                             hspace=0.5, wspace=0.1)
        
        # 4. Enhanced Summary Statistics (bottom right)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_summary_stats(ax1)

        # 1. Tool Calls Histogram (top, spanning columns)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_tool_calls_histogram(ax2)
        
        # 2. API Calls per Agent (top right)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_api_calls(ax3)
        
        # 3. Execution Flow Diagram (bottom left and middle)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_execution_flow(ax4)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.show()
        plt.close()
        
        print(f"Analysis saved to: {output_path}")
    
    def _plot_tool_calls_histogram(self, ax):
        """Plot histogram of tool calls by agent."""
        # Prepare data
        tool_data = defaultdict(Counter)
        for agent, tools in self.tool_calls.items():
            for tool in tools:
                tool_data[agent][tool] += 1
        
        # Get all unique tools
        all_tools = set()
        for tools in tool_data.values():
            all_tools.update(tools.keys())
        all_tools = sorted(list(all_tools))
        
        # Create grouped bar chart
        agents = ['Orchestrator', 'Research Agent', 'Analysis Agent', 'Writer Agent']
        x = np.arange(len(all_tools))
        width = 0.2
        
        # Purple gradient colors - more distinct shades + one for Orchestrator
        colors = {
            'Orchestrator': '#8B008B',      # Dark Magenta for Orchestrator
            'Research Agent': '#DDA0DD',    # Plum
            'Analysis Agent': '#9b59b6',    # Medium Purple
            'Writer Agent': '#4B0082'       # Indigo
        }
        
        for i, agent in enumerate(agents):
            counts = [tool_data[agent][tool] for tool in all_tools]
            offset = (i - len(agents)/2 + 0.5) * width
            color = colors.get(agent, '#999999')
            label = agent.replace(' Agent', '') if agent != 'Orchestrator' else agent
            bars = ax.bar(x + offset, counts, width, label=label, 
                          color=color, alpha=0.8)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Tool/Action', fontsize=12)
        ax.set_ylabel('Number of Calls', fontsize=12)
        ax.set_title('Tool Calls by Agent', fontsize=14, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(all_tools, rotation=15, ha='right')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        if tool_data:
            ax.set_ylim(0, max(count for counts in tool_data.values() for count in counts.values()) + 1)
    
    def _plot_api_calls(self, ax):
        """Plot API calls (response generations) per agent."""
        agents = ["Orchestrator", "Research Agent", "Analysis Agent", "Writer Agent"]
        counts = [self.api_calls.get(agent, 0) for agent in agents]
        
        # Purple gradient colors - more distinct shades
        colors = {
            'Orchestrator': '#8B008B',      # Dark Magenta
            'Research Agent': '#DDA0DD',    # Plum
            'Analysis Agent': '#9b59b6',    # Medium Purple
            'Writer Agent': '#4B0082'       # Indigo
        }
        
        bar_colors = [colors[agent] for agent in agents]
        bars = ax.bar(range(len(agents)), counts, color=bar_colors,
                      alpha=0.9, edgecolor='white', linewidth=2)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom', fontsize=11)
        
        ax.set_xticks(range(len(agents)))
        ax.set_xticklabels([a.replace(' Agent', '') if a != 'Orchestrator' else a for a in agents], 
                           rotation=15, fontsize=10)
        ax.set_ylabel('Number of API Calls', fontsize=12)
        ax.set_title('API Calls per Agent', fontsize=14, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(counts) + 2 if counts else 5)
        
        # Add subtle background
        ax.set_facecolor('#f8f9fa')
    
    def _plot_execution_flow(self, ax):
        """Plot the execution flow showing orchestrator coordination."""
        # Count transitions
        transitions = defaultdict(int)
        for i in range(len(self.execution_flow) - 1):
            key = (self.execution_flow[i], self.execution_flow[i + 1])
            transitions[key] += 1
        
        # Define positions for agents - Orchestrator on top row
        agent_positions = {
            'Orchestrator': (0.5, 0.8),      # Top center
            'Research Agent': (0, 0.3),      # Bottom left
            'Analysis Agent': (0.5, 0.3),    # Bottom center
            'Writer Agent': (1, 0.3)         # Bottom right
        }
        
        # Purple gradient colors
        agent_colors = {
            'Orchestrator': '#8B008B',       # Dark Magenta
            'Research Agent': '#DDA0DD',     # Plum
            'Analysis Agent': '#9b59b6',     # Medium Purple
            'Writer Agent': '#4B0082'        # Indigo
        }
        
        # Draw agents as boxes
        for agent, (x, y) in agent_positions.items():
            if agent in self.agents:
                color = agent_colors.get(agent, '#gray')
                
                # Make Orchestrator box slightly larger
                box_width = 0.26 if agent == 'Orchestrator' else 0.24
                box_height = 0.26 if agent == 'Orchestrator' else 0.24
                
                box = FancyBboxPatch((x-box_width/2, y-box_height/2), box_width, box_height,
                                    boxstyle="round,pad=0.03",
                                    facecolor=color, edgecolor='white',
                                    linewidth=3, alpha=0.9)
                ax.add_patch(box)
                
                # Adjust text for agent names
                display_name = agent if agent == 'Orchestrator' else agent.replace(' Agent', '')
                font_size = 13 if agent == 'Orchestrator' else 12
                
                ax.text(x, y, display_name, 
                       ha='center', va='center', fontweight='bold',
                       color='white', fontsize=font_size)
        
        # Draw arrows for transitions
        for (from_agent, to_agent), count in transitions.items():
            if from_agent in agent_positions and to_agent in agent_positions:
                x1, y1 = agent_positions[from_agent]
                x2, y2 = agent_positions[to_agent]
                
                # Calculate arrow properties
                if from_agent == to_agent:
                    # Self loop
                    ax.annotate('', xy=(x1, y1+0.15), xytext=(x1, y1+0.15),
                               arrowprops=dict(arrowstyle='->', 
                                             connectionstyle="arc3,rad=.8",
                                             color='#7f8c8d', lw=2+count/2,
                                             alpha=0.7))
                else:
                    # Adjust arrow endpoints based on box sizes
                    box_offset = 0.13 if from_agent == 'Orchestrator' or to_agent == 'Orchestrator' else 0.12
                    
                    # Calculate direction
                    dx = x2 - x1
                    dy = y2 - y1
                    angle = np.arctan2(dy, dx)
                    
                    # Adjust start and end points
                    start_x = x1 + box_offset * np.cos(angle)
                    start_y = y1 + box_offset * np.sin(angle)
                    end_x = x2 - box_offset * np.cos(angle)
                    end_y = y2 - box_offset * np.sin(angle)
                    
                    ax.annotate('', xy=(end_x, end_y), 
                               xytext=(start_x, start_y),
                               arrowprops=dict(arrowstyle='->', 
                                             color='#7f8c8d', 
                                             lw=2+count/2,
                                             alpha=0.7))
                    
                    # Add count label
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    ax.text(mid_x, mid_y, str(count), 
                           ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.3", 
                                   facecolor='white', 
                                   edgecolor='#7f8c8d',
                                   alpha=0.9),
                           fontsize=11, fontweight='bold')
        
        ax.set_xlim(-0.25, 1.25)
        ax.set_ylim(0.1, 1)
        ax.set_title(f'Execution Flow Diagram', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_facecolor('#f8f9fa')
        ax.axis('off')
    
    def _plot_summary_stats(self, ax):
        """Plot enhanced summary statistics with explanations."""
        total_tool_calls = sum(len(tools) for tools in self.tool_calls.values())
        
        # Create the main title
        title_text = "Performance Metrics"
        ax.text(0.5, 0.95, title_text, transform=ax.transAxes, 
               fontsize=14, fontweight='bold', ha='center', va='top')
        
        # Create metrics content with formatting
        metrics = [
            (f"Runtime: {self._calculate_runtime()} seconds", 
             "The total time taken for the task to complete"),
            (f"Total Agents: {len(self.agents)}", 
             "The number of unique agents that participated in this task"),
            
            (f"Total Tool Calls: {total_tool_calls}", 
             "Actions taken by agents (web_search, save_to_memory, etc.)"),
            
            (f"Total API Calls: {sum(self.api_calls.values())}", 
             "Number of times agents generated responses (LLM calls)"),
            
            (f"Flow Changes: {len(self.execution_flow) - 1}", 
             "Number of handoffs between agents during task execution"),
        ]
        
        # Display metrics with proper formatting
        y_pos = 0.80
        for metric, description in metrics:
            # Metric value (bold, normal size)
            ax.text(0.05, y_pos, metric, transform=ax.transAxes, 
                   fontsize=11, va='top')
            # Description (smaller, lighter)
            ax.text(0.05, y_pos - 0.05, description, transform=ax.transAxes, 
                   fontsize=9, color='#666666', va='top')
            y_pos -= 0.14
        
        # Add a subtle border
        rect = plt.Rectangle((0.02, 0.02), 0.96, 0.96,
                           transform=ax.transAxes, 
                           facecolor='#9b59b6', 
                           edgecolor='#4B0082', 
                           linewidth=2,
                           alpha=0.3)
        ax.add_patch(rect)
        
        ax.axis('off')

    def _calculate_runtime(self):
        """Calculate the runtime in minutes and seconds."""
        if self.start_time and self.end_time:
            # Parse times
            start_parts = self.start_time.split(':')
            end_parts = self.end_time.split(':')
            
            # Convert to seconds
            start_seconds = int(start_parts[0]) * 3600 + int(start_parts[1]) * 60 + int(start_parts[2])
            end_seconds = int(end_parts[0]) * 3600 + int(end_parts[1]) * 60 + int(end_parts[2])
            
            # Calculate difference
            diff_seconds = end_seconds - start_seconds
            
            # Handle case where task runs past midnight
            if diff_seconds < 0:
                diff_seconds += 24 * 3600
            
            return diff_seconds
        return 0

def main():
    # Get log file path from command line or use default
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        # Find the most recent log file
        logs_dir = Path('logs')
        if not logs_dir.exists():
            print("Error: logs directory not found")
            sys.exit(1)
        
        log_files = list(logs_dir.glob('orchestrator_log_*.txt'))
        if not log_files:
            print("Error: No orchestrator team log files found")
            sys.exit(1)
        
        log_path = max(log_files, key=lambda p: p.stat().st_mtime)
    
    # Read log content
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
    except Exception as e:
        print(f"Error reading log file: {e}")
        sys.exit(1)
    
    # Create analyzer and parse log
    analyzer = OrchestratorLogAnalyzer(log_content)
    analyzer.parse_log()
    
    # Create output directory if it doesn't exist
    analysis_dir = Path('analysis')
    analysis_dir.mkdir(exist_ok=True)
    
    # Generate output filename based on task name
    safe_task_name = re.sub(r'[^\w\s-]', '', analyzer.task_name)
    safe_task_name = re.sub(r'[-\s]+', '_', safe_task_name)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'orchestrator_analysis_{timestamp}_{safe_task_name}.png'
    output_path = analysis_dir / output_filename
    
    # Create and save visualization
    analyzer.create_visualization(output_path)

if __name__ == "__main__":
    main()