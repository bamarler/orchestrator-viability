from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from typing import Dict, Optional
import json

# Initialize search tool
search = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    """Search the internet using DuckDuckGo."""
    try:
        return search.invoke(query)
    except Exception as e:
        return f"Search error: {str(e)}"

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        # Basic safety check
        allowed_chars = "0123456789+-*/()., "
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        return str(eval(expression))
    except Exception as e:
        return f"Calculation error: {str(e)}"

@tool
def save_to_memory(input: str) -> str:
    """Save information to shared memory. Format: 'key::value'"""
    try:
        key, value = input.split('::', 1)
    except ValueError:
        return "Error: Input must be in format 'key::value'"
    try:
        with open('shared_memory.json', 'r') as f:
            memory = json.load(f)
    except:
        memory = {}
    
    memory[key] = value
    
    with open('shared_memory.json', 'w') as f:
        json.dump(memory, f)
    
    return f"Saved {key} to memory"

@tool
def read_from_memory(key: str) -> str:
    """Read information from shared memory."""
    try:
        with open('shared_memory.json', 'r') as f:
            memory = json.load(f)
        return memory.get(key, f"No data found for key: {key}")
    except:
        return "Memory is empty"

@tool
def list_memory_keys() -> str:
    """List all keys in shared memory."""
    try:
        with open('shared_memory.json', 'r') as f:
            memory = json.load(f)
        keys = list(memory.keys())
        return f"Memory keys: {', '.join(keys)}" if keys else "Memory is empty"
    except:
        return "Memory is empty"

@tool
def extract_facts(text: str) -> str:
    """Extract key facts from a text passage."""
    # Simple fact extraction - in production, use NLP
    sentences = text.split('.')
    facts = []
    
    # Look for sentences with numbers, dates, or key phrases
    for sent in sentences:
        sent = sent.strip()
        if any(char.isdigit() for char in sent) or \
           any(keyword in sent.lower() for keyword in ['first', 'largest', 'most', 'important', 'key']):
            facts.append(sent)
    
    return '\n'.join(facts[:5]) if facts else "No key facts found"

@tool
def word_count(text: str) -> str:
    """Count the number of words in a text."""
    words = len(text.split())
    return f"Word count: {words}"

@tool
def check_factual_consistency(claims: str) -> str:
    """Check if two claims are consistent with each other. Include claims in format 'claim1::claim2'"""
    # In production, use semantic similarity or LLM
    # For now, simple keyword overlap
    try:
        claim1, claim2 = claims.split('::', 1)
    except ValueError:
        return "Error: Input must be in format 'claim1::claim2'"
    words1 = set(claim1.lower().split())
    words2 = set(claim2.lower().split())
    
    overlap = len(words1.intersection(words2))
    total = len(words1.union(words2))
    
    similarity = overlap / total if total > 0 else 0
    
    if similarity > 0.7:
        return "Claims appear consistent"
    elif similarity > 0.3:
        return "Claims have some overlap but may differ in details"
    else:
        return "Claims appear to be inconsistent or about different topics"

@tool
def create_outline(input: str) -> str:
    """Create a comprehensive outline for a report. Format: 'topic' or 'topic::body_sections' (default 3 body sections)"""
    try:
        parts = input.split('::', 1)
        topic = parts[0]
        body_sections = int(parts[1]) if len(parts) > 1 else 3  # Default to 3 body sections
    except (ValueError, IndexError):
        topic = input
        body_sections = 3  # Default to 3 body sections
    
    # Ensure minimum of 3 body sections
    body_sections = max(body_sections, 3)
    
    outline = f"Report Outline: {topic}\n\n"
    outline += "1. Abstract\n"
    outline += "2. Introduction\n"
    
    for i in range(body_sections):
        section_num = i + 3  # Starting after Abstract(1) and Introduction(2)
        outline += f"{section_num}. [Body Section {i + 1}]\n"
    
    outline += f"{body_sections + 3}. Conclusion and Recommendations\n"
    
    return outline

@tool
def format_citation(source: str) -> str:
    """Format a citation for a source. Optionally include URL in format 'source::url'"""
    try:
        source, url = source.split('::', 1)
        return f"[{source}]({url})"
    except ValueError:
        pass
    return f"Source: {source}"

# Tool collections for different agent types
RESEARCH_TOOLS = [web_search, save_to_memory, extract_facts, format_citation, create_outline, read_from_memory, list_memory_keys]
ANALYSIS_TOOLS = [calculator, read_from_memory, save_to_memory, list_memory_keys, extract_facts, check_factual_consistency]
WRITER_TOOLS = [read_from_memory, save_to_memory, list_memory_keys, word_count, create_outline]
ORCHESTRATOR_TOOLS = [read_from_memory, list_memory_keys]

# All tools available
ALL_TOOLS = [
    web_search,
    calculator,
    save_to_memory,
    read_from_memory,
    list_memory_keys,
    extract_facts,
    word_count,
    check_factual_consistency,
    create_outline,
    format_citation
]