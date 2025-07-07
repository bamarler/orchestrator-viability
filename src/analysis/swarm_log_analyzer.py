#!/usr/bin/env python3
"""
Swarm Team Performance Analyzer
Analyzes swarm team logs and creates visualizations of team performance metrics.
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

class SwarmLogAnalyzer:
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
            if "Starting swarm team for task:" in line:
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
            
            # Extract agent name
            agent_match = re.search(r'(\w+ Agent) \(Swarm Team\):', line)
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
        main_title = f'Swarm Log Analysis: {self.task_name}'
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
    
    def _plot_communication_matrix(self, ax):
        """Plot agent-to-agent communication matrix."""
        agents = sorted(list(self.agents))
        matrix = np.zeros((len(agents), len(agents)))
        
        for i, from_agent in enumerate(agents):
            for j, to_agent in enumerate(agents):
                matrix[i, j] = self.communications[from_agent][to_agent]
        
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(agents)))
        ax.set_yticks(np.arange(len(agents)))
        ax.set_xticklabels([a.replace(' Agent', '') for a in agents], rotation=45)
        ax.set_yticklabels([a.replace(' Agent', '') for a in agents])
        
        # Add text annotations
        for i in range(len(agents)):
            for j in range(len(agents)):
                if matrix[i, j] > 0:
                    ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")
        
        ax.set_title('Agent Communication Matrix\n(From → To)', fontweight='bold')
        ax.set_xlabel('To Agent')
        ax.set_ylabel('From Agent')
    
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
        agents = sorted(list(tool_data.keys()))
        x = np.arange(len(all_tools))
        width = 0.25
        
        # Purple gradient colors - more distinct shades
        colors = ['#DDA0DD', '#9b59b6', '#4B0082']  # Plum, Medium Purple, Indigo
        
        for i, agent in enumerate(["Research Agent", "Analysis Agent", "Writer Agent"]):
            counts = [tool_data[agent][tool] for tool in all_tools]
            offset = (i - len(agents)/2 + 0.5) * width
            bars = ax.bar(x + offset, counts, width, label=agent.replace(' Agent', ''), 
                          color=colors[i % len(colors)], alpha=0.8)
            
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
        ax.set_ylim(0, max(count for counts in tool_data.values() for count in counts.values()) + 1)
    
    def _plot_api_calls(self, ax):
        """Plot API calls (response generations) per agent."""
        agents = ["Research Agent", "Analysis Agent", "Writer Agent"]
        counts = [self.api_calls[agent] for agent in agents]
        
        # Purple gradient colors - more distinct shades
        colors = ['#DDA0DD', '#9b59b6', '#4B0082']  # Plum, Medium Purple, Indigo
        bars = ax.bar(range(len(agents)), counts, color=[colors[i % len(colors)] for i in range(len(agents))],
                      alpha=0.9, edgecolor='white', linewidth=2)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom', fontsize=11)
        
        ax.set_xticks(range(len(agents)))
        ax.set_xticklabels([a.replace(' Agent', '') for a in agents], rotation=0, fontsize=11)
        ax.set_ylabel('Number of API Calls', fontsize=12)
        ax.set_title('API Calls per Agent', fontsize=14, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(counts) + 2)
        
        # Add subtle background
        ax.set_facecolor('#f8f9fa')
    
    def _plot_execution_flow(self, ax):
        """Plot the execution flow showing linearity."""
        # Count transitions
        transitions = defaultdict(int)
        for i in range(len(self.execution_flow) - 1):
            key = (self.execution_flow[i], self.execution_flow[i + 1])
            transitions[key] += 1
        
        # Define positions for agents
        agent_positions = {
            'Research Agent': (0, 0.5),
            'Analysis Agent': (0.5, 0.5),
            'Writer Agent': (1, 0.5)
        }
        
        # Purple gradient colors - more distinct shades
        agent_colors = {
            'Research Agent': '#DDA0DD',  # Plum
            'Analysis Agent': '#9b59b6',  # Medium Purple
            'Writer Agent': '#4B0082'     # Indigo
        }
        
        # Draw agents as boxes
        for agent, (x, y) in agent_positions.items():
            if agent in self.agents:
                color = agent_colors.get(agent, '#gray')
                
                box = FancyBboxPatch((x-0.12, y-0.12), 0.24, 0.24,
                                    boxstyle="round,pad=0.03",
                                    facecolor=color, edgecolor='white',
                                    linewidth=3, alpha=0.9)
                ax.add_patch(box)
                ax.text(x, y, agent.replace(' Agent', ''), 
                       ha='center', va='center', fontweight='bold',
                       color='white', fontsize=12)
        
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
                    # Adjust arrow position based on direction
                    offset = 0.03 if x2 > x1 else -0.03
                    ax.annotate('', xy=(x2-0.12, y2+offset), 
                               xytext=(x1+0.12, y1+offset),
                               arrowprops=dict(arrowstyle='->', 
                                             color='#7f8c8d', 
                                             lw=2+count/2,
                                             alpha=0.7))
                    
                    # Add count label
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2 + offset
                    ax.text(mid_x, mid_y, str(count), 
                           ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.3", 
                                   facecolor='white', 
                                   edgecolor='#7f8c8d',
                                   alpha=0.9),
                           fontsize=11, fontweight='bold')
        
        # Calculate linearity score
        linear_count = 0
        total_transitions = sum(transitions.values())
        
        for (from_agent, to_agent), count in transitions.items():
            if (from_agent == 'Research Agent' and to_agent == 'Analysis Agent') or \
               (from_agent == 'Analysis Agent' and to_agent == 'Writer Agent'):
                linear_count += count
        
        linearity_score = (linear_count / max(1, total_transitions)) * 100 if total_transitions > 0 else 100
        
        ax.set_xlim(-0.25, 1.25)
        ax.set_ylim(0.2, 1)
        ax.set_title(f'Execution Flow Diagram  (Linearity Score: {linearity_score:.1f}%)', 
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
        
        log_files = list(logs_dir.glob('swarm_log_*.txt'))
        if not log_files:
            print("Error: No swarm team log files found")
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
    analyzer = SwarmLogAnalyzer(log_content)
    analyzer.parse_log()
    
    # Create output directory if it doesn't exist
    analysis_dir = Path('analysis')
    analysis_dir.mkdir(exist_ok=True)
    
    # Generate output filename based on task name
    safe_task_name = re.sub(r'[^\w\s-]', '', analyzer.task_name)
    safe_task_name = re.sub(r'[-\s]+', '_', safe_task_name)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'swarm_analysis_{timestamp}_{safe_task_name}.png'
    output_path = analysis_dir / output_filename
    
    # Create and save visualization
    analyzer.create_visualization(output_path)

if __name__ == "__main__":
    main()