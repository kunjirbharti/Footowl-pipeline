from langgraph.graph import StateGraph, START, END
from typing import Dict, Any

from models import AgentState
from rag import SimpleRAG
from agents import (
    agent_parse_intent,
    agent_analyze_images,
    agent_write_storyboard,
    agent_generate_script,
    agent_compile_and_fix,
    agent_render
)

def build_pipeline_graph(project_dir: str, rag: SimpleRAG) -> StateGraph:
    """Builds and compiles the LangGraph StateGraph representing the multiagent pipeline."""
    
    workflow = StateGraph(AgentState)
    
    # 1. Define nodes mapping state transitions to agents
    def parse_intent_node(state: AgentState) -> Dict[str, Any]:
        return agent_parse_intent(state)
        
    def analyze_images_node(state: AgentState) -> Dict[str, Any]:
        return agent_analyze_images(state)
        
    def write_storyboard_node(state: AgentState) -> Dict[str, Any]:
        return agent_write_storyboard(state, rag)
        
    def generate_script_node(state: AgentState) -> Dict[str, Any]:
        return agent_generate_script(state, rag)
        
    def compile_and_fix_node(state: AgentState) -> Dict[str, Any]:
        return agent_compile_and_fix(state, project_dir)
        
    def render_node(state: AgentState) -> Dict[str, Any]:
        return agent_render(state, project_dir)
        
    # 2. Add nodes to graph
    workflow.add_node("parse_intent", parse_intent_node)
    workflow.add_node("analyze_images", analyze_images_node)
    workflow.add_node("write_storyboard", write_storyboard_node)
    workflow.add_node("generate_script", generate_script_node)
    workflow.add_node("compile_and_fix", compile_and_fix_node)
    workflow.add_node("render", render_node)
    
    # 3. Define static transitions
    workflow.add_edge(START, "parse_intent")
    workflow.add_edge("parse_intent", "analyze_images")
    workflow.add_edge("analyze_images", "write_storyboard")
    workflow.add_edge("write_storyboard", "generate_script")
    workflow.add_edge("generate_script", "compile_and_fix")
    
    # 4. Define conditional transitions for the retry loop
    def routing_condition(state: AgentState) -> str:
        status = state.get("status", "")
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)
        
        if status == "compiled":
            print("[Graph router] Compilation succeeded. Routing to Renderer.")
            return "render"
        elif status == "compile_failed" and retry_count < max_retries:
            print(f"[Graph router] Compilation failed (Attempt {retry_count}/{max_retries}). Routing back to Script Generator.")
            return "generate_script"
        else:
            print("[Graph router] Compilation failed and retry limit reached. Exiting gracefully.")
            return END
            
    workflow.add_conditional_edges(
        "compile_and_fix",
        routing_condition,
        {
            "render": "render",
            "generate_script": "generate_script",
            END: END
        }
    )
    
    # Render completes the graph
    workflow.add_edge("render", END)
    
    return workflow.compile()
