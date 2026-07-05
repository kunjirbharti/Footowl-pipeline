import os
import sys
import argparse
import json
import glob
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from rag import SimpleRAG
from graph import build_pipeline_graph

def run(images_dir: str, prompt: str, max_retries: int = 3, output_dir: str = "sample_output"):
    """Runs the Image-to-Video Multiagent Pipeline on a directory of images with a prompt."""
    print("=" * 60)
    print("Starting FotoOwl Image-to-Video Multiagent Pipeline")
    print("=" * 60)
    print(f"Images Directory: {images_dir}")
    print(f"User Prompt:      {prompt}")
    print(f"Max Retries:      {max_retries}")
    print("-" * 60)

    # 1. Validate images directory
    if not os.path.exists(images_dir):
        print(f"Error: Images directory '{images_dir}' does not exist.")
        sys.exit(1)
        
    # Supported image formats
    image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    image_paths = []
    for pattern in image_patterns:
        # Match case-insensitively
        image_paths.extend(glob.glob(os.path.join(images_dir, pattern)))
        image_paths.extend(glob.glob(os.path.join(images_dir, pattern.upper())))
        
    image_paths = sorted(list(set(image_paths)))
    
    if not image_paths:
        print(f"Error: No images (*.jpg, *.jpeg, *.png, *.webp) found in '{images_dir}'.")
        sys.exit(1)
        
    print(f"Found {len(image_paths)} images to process.")
    
    # 2. Setup project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 3. Initialize RAG
    # Set use_gemini_embeddings=True if GEMINI_API_KEY is available
    use_gemini_embeddings = bool(os.getenv("GEMINI_API_KEY"))
    print(f"Initializing RAG store (Gemini Embeddings: {use_gemini_embeddings})...")
    rag = SimpleRAG(use_gemini_embeddings=use_gemini_embeddings)
    
    # 4. Compile the LangGraph
    print("Compiling LangGraph Agent StateGraph...")
    graph = build_pipeline_graph(project_dir, rag)
    
    # 5. Run the pipeline
    initial_state = {
        "user_prompt": prompt,
        "image_paths": image_paths,
        "retry_count": 0,
        "max_retries": max_retries,
        "compile_errors": [],
        "status": "started"
    }
    
    print("\nRunning pipeline agents...")
    consolidated_state = graph.invoke(initial_state)
    
    print("\n" + "=" * 60)
    print("Pipeline Execution Complete!")
    print("=" * 60)
    print(f"Final Status: {consolidated_state.get('status')}")
    print(f"Compiler Retries: {consolidated_state.get('retry_count')} / {max_retries}")
    print(f"Rendered Video: {consolidated_state.get('output_video_path')}")
    print("-" * 60)
    
    # 6. Save outputs to sample_output folder
    out_path = os.path.join(project_dir, output_dir)
    os.makedirs(out_path, exist_ok=True)
    
    # Save Storyboard JSON
    storyboard = consolidated_state.get("storyboard")
    if storyboard:
        sb_file = os.path.join(out_path, "storyboard.json")
        with open(sb_file, "w", encoding="utf-8") as f:
            f.write(storyboard.model_dump_json(indent=2))
        print(f"Saved Storyboard JSON to: {sb_file}")
        
    # Save Generated TSX Script
    script_code = consolidated_state.get("remotion_script")
    if script_code:
        script_file = os.path.join(out_path, "Slideshow.tsx")
        with open(script_file, "w", encoding="utf-8") as f:
            f.write(script_code)
        print(f"Saved Remotion React component to: {script_file}")
        
    # Save Final Pipeline State
    # Convert Pydantic models to dicts for JSON serialization
    serialized_state = {}
    for k, v in consolidated_state.items():
        if hasattr(v, "model_dump"):
            serialized_state[k] = v.model_dump()
        elif isinstance(v, list):
            serialized_list = []
            for item in v:
                if hasattr(item, "model_dump"):
                    serialized_list.append(item.model_dump())
                else:
                    serialized_list.append(item)
            serialized_state[k] = serialized_list
        else:
            serialized_state[k] = v
            
    state_file = os.path.join(out_path, "pipeline_state.json")
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(serialized_state, f, indent=2)
    print(f"Saved final pipeline state trace to: {state_file}")
    
    print("=" * 60)
    return consolidated_state

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FotoOwl AI Image-to-Video Pipeline CLI")
    parser.add_argument("--images_dir", required=True, help="Directory containing event photos")
    parser.add_argument("--prompt", required=True, help="Creative brief prompt defining video style")
    parser.add_argument("--max_retries", type=int, default=3, help="Max compilation fix retries")
    parser.add_argument("--output_dir", default="sample_output", help="Directory to save execution traces")
    
    args = parser.parse_args()
    run(args.images_dir, args.prompt, args.max_retries, args.output_dir)
