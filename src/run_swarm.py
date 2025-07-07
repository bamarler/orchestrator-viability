"""
Run the swarm team directly from src directory.
Usage: python run_swarm.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.swarm.swarm_graph import run_swarm_team

if __name__ == "__main__":
    # You can modify the task here
    task = "renewable energy trends in 2024"
    word_count = 1000
    
    print(f"Running Swarm Team")
    print(f"Task: {task}")
    print(f"Target word count: {word_count}")
    print("="*60)
    
    report = run_swarm_team(task, word_count=word_count)
    
    print("\nSwarm Team Test Complete!")
    print("="*50)
    print("\nGenerated files:")
    print("- Check 'reports' directory for the full report")
    print("- Check 'logs' directory for the detailed peer-to-peer communication log")
    print("\nReport Preview:")
    print("-"*50)
    if report:
        print(report[:500] + "..." if len(report) > 500 else report)
    else:
        print("No report generated - check logs for errors")