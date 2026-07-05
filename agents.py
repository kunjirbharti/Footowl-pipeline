import os
import shutil
import subprocess
import json
import re
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field
from PIL import Image

from models import VideoIntent, ImageAnalysis, StoryboardSlide, Storyboard, AgentState
from rag import SimpleRAG

# Standard helper to call Gemini structured outputs
def call_gemini_structured(model_name: str, prompt: str, response_schema: Any, image: Optional[Image.Image] = None) -> Any:
    """Helper to make a structured API call to Gemini."""
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(model_name)
    
    contents = [prompt]
    if image is not None:
        contents.append(image)
        
    generation_config = genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema=response_schema,
        temperature=0.2
    )
    
    response = model.generate_content(
        contents,
        generation_config=generation_config
    )
    
    # Parse the response JSON
    try:
        return response_schema.model_validate_json(response.text)
    except Exception as e:
        # Fallback to direct json loading if model validation fails on wrapper
        data = json.loads(response.text)
        return response_schema(**data)

# Helper to check if Node/NPM is available in path
def check_node_available() -> bool:
    try:
        subprocess.run(["node", "-v"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        subprocess.run(["npm", "-v"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

# Pydantic schemas for intermediate LLM calls that aren't in models.py
class ImageSelectionList(BaseModel):
    selected_paths: List[str] = Field(description="A sorted list of the selected image absolute paths representing the final sequence")

class GeneratedScript(BaseModel):
    code: str = Field(description="The complete, valid React TypeScript code for Slideshow.tsx starting with imports and ending with exports. No markdown backticks.")


# 1. Intent Parser Agent
def agent_parse_intent(state: AgentState) -> Dict[str, Any]:
    user_prompt = state["user_prompt"]
    print(f"[Intent Parser] Parsing prompt: '{user_prompt}'")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        # Mock Intent Parsing
        print("[Intent Parser] WARNING: No GEMINI_API_KEY found. Running in MOCK mode.")
        prompt_lower = user_prompt.lower()
        if "wedding" in prompt_lower or "cinematic" in prompt_lower or "slow" in prompt_lower:
            intent = VideoIntent(
                pacing="slow",
                visual_style="cinematic",
                caption_tone="emotional",
                transition_preference="fade",
                font_style="serif"
            )
        elif "birthday" in prompt_lower or "upbeat" in prompt_lower or "fast" in prompt_lower or "energetic" in prompt_lower:
            intent = VideoIntent(
                pacing="fast",
                visual_style="upbeat",
                caption_tone="bold",
                transition_preference="zoom",
                font_style="bold-display"
            )
        else: # Default corporate
            intent = VideoIntent(
                pacing="medium",
                visual_style="corporate",
                caption_tone="professional",
                transition_preference="slide",
                font_style="sans-serif"
            )
        return {"video_intent": intent}
        
    prompt = f"Parse the following video request prompt into a structured video intent: '{user_prompt}'"
    intent = call_gemini_structured("gemini-1.5-flash", prompt, VideoIntent)
    print(f"[Intent Parser] Result: {intent}")
    return {"video_intent": intent}


# 2. Image Analyser Agent
def agent_analyze_images(state: AgentState) -> Dict[str, Any]:
    image_paths = state["image_paths"]
    intent = state["video_intent"]
    print(f"[Image Analyser] Analyzing {len(image_paths)} images...")
    
    api_key = os.getenv("GEMINI_API_KEY")
    analyses = []
    
    if not api_key:
        print("[Image Analyser] WARNING: No GEMINI_API_KEY found. Running in MOCK mode.")
        # Generate mock analyses that align with the intent
        style = intent.visual_style
        for i, path in enumerate(image_paths):
            basename = os.path.basename(path)
            # Create a mock description and details
            if style == "cinematic":
                subject = f"A beautiful romantic moment captured in {basename}, featuring warm ambient light"
                palette = ["warm amber", "rose gold", "champagne cream"]
                mood = "romantic and emotional"
                caption = "A day filled with endless love" if i == 0 else f"Moments we will cherish forever ({i+1})"
            elif style == "upbeat":
                subject = f"An exciting action shot in {basename}, showing high energy and celebration"
                palette = ["vivid yellow", "electric blue", "neon pink"]
                mood = "joyful and high-energy"
                caption = "HAPPY BIRTHDAY!" if i == 0 else f"Party vibes on point! 🎉"
            else:
                subject = f"A professional corporate highlight in {basename}, featuring sharp focus and clean lines"
                palette = ["navy blue", "steel grey", "stark white"]
                mood = "professional and ambitious"
                caption = "Innovating the future of technology" if i == 0 else f"Collaborating for excellence"
                
            analyses.append(ImageAnalysis(
                image_path=path,
                subject=subject,
                color_palette=palette,
                mood=mood,
                quality_score=9.0 - (i % 3) * 0.5, # Varying quality score
                recommended_caption=caption
            ))
    else:
        # Real vision analysis
        for path in image_paths:
            print(f"[Image Analyser] Processing: {os.path.basename(path)}")
            try:
                img = Image.open(path)
                # Resize image to save bandwidth/tokens
                img.thumbnail((512, 512))
                
                prompt = (
                    "Analyze this image for an automated video slideshow. "
                    "Determine the main subject, dominant colors, emotional mood, a composition quality score (1.0 to 10.0), "
                    "and write a short recommended caption."
                )
                
                # We request ImageAnalysis but the schema doesn't match path. We can wrap it.
                class VisionResult(BaseModel):
                    subject: str
                    color_palette: List[str]
                    mood: str
                    quality_score: float
                    recommended_caption: str
                    
                res = call_gemini_structured("gemini-1.5-flash", prompt, VisionResult, image=img)
                analyses.append(ImageAnalysis(
                    image_path=path,
                    subject=res.subject,
                    color_palette=res.color_palette,
                    mood=res.mood,
                    quality_score=res.quality_score,
                    recommended_caption=res.recommended_caption
                ))
            except Exception as e:
                print(f"[Image Analyser] Error analyzing {path}: {e}")
                # Fallback on failure
                analyses.append(ImageAnalysis(
                    image_path=path,
                    subject="Image parsing failed",
                    color_palette=["unknown"],
                    mood="neutral",
                    quality_score=5.0,
                    recommended_caption="A beautiful memory"
                ))

    # Selection logic: Select best subset of images (discard worst, limit to max 8 images for reel)
    # Sort by quality score, filter out extremely low scores, then select a subset
    sorted_analyses = sorted(analyses, key=lambda x: x.quality_score, reverse=True)
    
    # We select up to 8 images
    selected_analyses = sorted_analyses[:8]
    # Maintain the original sorting order of the files so sequencing makes chronological sense
    selected_analyses.sort(key=lambda x: image_paths.index(x.image_path))
    selected_images = [a.image_path for a in selected_analyses]
    
    print(f"[Image Analyser] Selected {len(selected_images)} out of {len(image_paths)} images.")
    return {
        "image_analyses": analyses,
        "selected_images": selected_images
    }


# 3. Storyboard Writer Agent
def agent_write_storyboard(state: AgentState, rag: SimpleRAG) -> Dict[str, Any]:
    intent = state["video_intent"]
    analyses = state["image_analyses"]
    selected_images = state["selected_images"]
    
    print("[Storyboard Writer] Retrieving style guides from RAG...")
    # Retrieve style guide snippet from RAG
    style_results = rag.retrieve("style_guides", query=intent.visual_style, limit=1)
    style_context = style_results[0]["content"] if style_results else "No style guide found."
    print(f"[Storyboard Writer] Style guide retrieved: {style_results[0]['style_name'] if style_results else 'None'}")
    
    # Filter image analyses to only include selected images
    selected_analyses = [a for a in analyses if a.image_path in selected_images]
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[Storyboard Writer] WARNING: No GEMINI_API_KEY found. Running in MOCK mode.")
        # Generate mock storyboard based on retrieved style
        slides = []
        pacing_sec = 5.0 if intent.pacing == "slow" else (1.5 if intent.pacing == "fast" else 3.0)
        trans_type = intent.transition_preference
        trans_sec = 1.0 if intent.pacing == "slow" else (0.3 if intent.pacing == "fast" else 0.8)
        
        font = "Playfair Display" if intent.font_style == "serif" else ("Impact" if intent.font_style == "bold-display" else "Inter")
        size = "64px" if intent.font_style == "bold-display" else "48px"
        color = "#fdfbf7" if intent.visual_style == "cinematic" else ("#ffcc00" if intent.visual_style == "upbeat" else "#ffffff")
        
        for i, analysis in enumerate(selected_analyses):
            slides.append(StoryboardSlide(
                image_path=analysis.image_path,
                caption=analysis.recommended_caption,
                duration_sec=pacing_sec,
                transition_type="none" if i == 0 else trans_type,
                transition_duration_sec=0.0 if i == 0 else trans_sec,
                font_family=font,
                font_size=size,
                font_color=color
            ))
            
        storyboard = Storyboard(
            title=f"A Story in {intent.visual_style.capitalize()} Style",
            slides=slides
        )
        return {"storyboard": storyboard}
        
    prompt = (
        "You are a professional video director and storyboard editor. "
        "Create a structured storyboard by sequencing the selected images into a cohesive narrative arc. "
        f"\n\nStyle Guide Context: {style_context}"
        f"\n\nUser Pacing and Intent: {intent}"
        f"\n\nSelected Image Analyses: {selected_analyses}"
        "\n\nOutput a JSON object matching the Storyboard schema. Select matching transitions, fonts, captions, and timings."
    )
    
    storyboard = call_gemini_structured("gemini-1.5-pro", prompt, Storyboard)
    print(f"[Storyboard Writer] Storyboard written with {len(storyboard.slides)} slides.")
    return {"storyboard": storyboard}


# 4. Script Generator Agent
def agent_generate_script(state: AgentState, rag: SimpleRAG) -> Dict[str, Any]:
    storyboard = state["storyboard"]
    intent = state["video_intent"]
    errors = state.get("compile_errors", [])
    
    print("[Script Generator] Retrieving Remotion API snippets from RAG...")
    # Retrieve relevant API references
    # If there are errors, retrieve snippets related to the error. Otherwise, standard slideshow elements
    query_str = "useCurrentFrame, Composition, Sequence, Img, interpolate"
    if errors:
        query_str = f"Error: {errors[-1]}. Fix composition, imports, or React rendering."
        print(f"[Script Generator] Retrying with error context: {errors[-1][:100]}...")
        
    api_results = rag.retrieve("remotion_api", query=query_str, limit=3)
    api_context = "\n\n".join([r["content"] for r in api_results])
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[Script Generator] WARNING: No GEMINI_API_KEY found. Running in MOCK mode.")
        # Load the mock template or write a simple script
        # We will write a valid slideshow file by generating code
        slides_js = []
        for s in storyboard.slides:
            # Clean path backslashes for JS strings
            clean_path = s.image_path.replace("\\", "/")
            slides_js.append(
                f"  {{\n"
                f"    imagePath: '{clean_path}',\n"
                f"    caption: {json.dumps(s.caption)},\n"
                f"    durationFrames: {int(s.duration_sec * 30)},\n"
                f"    transitionFrames: {int(s.transition_duration_sec * 30)},\n"
                f"    transitionType: '{s.transition_type}',\n"
                f"    fontFamily: '{s.font_family}',\n"
                f"    fontSize: '{s.font_size}',\n"
                f"    fontColor: '{s.font_color}'\n"
                f"  }}"
            )
            
        slides_array_str = ",\n".join(slides_js)
        
        # Read the template Slideshow.tsx but replace the default slides
        # We can construct the complete TSX script
        code = get_template_slideshow_code(slides_array_str)
        return {"remotion_script": code}
        
    prompt = (
        "You are a specialized AI React and Remotion programmer. "
        "Write the complete React TypeScript code for a file named Slideshow.tsx. "
        "It must render a slideshow with the slides defined in the storyboard. "
        "\n\nStoryboard:\n"
        f"{storyboard.model_dump_json(indent=2)}"
        "\n\nRemotion API Reference Snippets:\n"
        f"{api_context}"
        "\n\nUser Creative Intent:\n"
        f"{intent.model_dump_json(indent=2)}"
        + (f"\n\nPrevious Compilation Errors to fix:\n{errors[-1]}" if errors else "") +
        "\n\nInstructions:\n"
        "1. Write complete TypeScript React code, importing React and necessary components from 'remotion'.\n"
        "2. Define the exact slides array inside the file using the storyboard slide data (convert seconds to frames at 30 FPS).\n"
        "3. Loop through slides and render them using <Sequence> with correctly calculated frame start offsets and durations.\n"
        "4. Implement transitions (fade, slide, zoom) and caption entries using interpolate(), spring(), and useCurrentFrame().\n"
        "5. Use the <Img> component from 'remotion' (or standard img if needed) for rendering.\n"
        "6. Do not include markdown wraps or explanations, output only the valid code inside the Pydantic 'code' property."
    )
    
    script_obj = call_gemini_structured("gemini-1.5-pro", prompt, GeneratedScript)
    print("[Script Generator] Generated Remotion script.")
    return {"remotion_script": script_obj.code}


# Helper template for mock mode script writing
def get_template_slideshow_code(slides_array_str: str) -> str:
    return f"""import React from 'react';
import {{ AbsoluteFill, Sequence, useCurrentFrame, useVideoConfig, interpolate, spring, Img }} from 'remotion';

interface Slide {{
  imagePath: string;
  caption: string;
  durationFrames: number;
  transitionFrames: number;
  transitionType: 'fade' | 'slide' | 'zoom' | 'none';
  fontFamily: string;
  fontSize: string;
  fontColor: string;
}}

const SLIDES: Slide[] = [
{slides_array_str}
];

export const Slideshow = () => {{
  const {{ fps }} = useVideoConfig();
  const frame = useCurrentFrame();

  // Calculate start frames for each slide sequence
  let currentStart = 0;
  const slideSequences = SLIDES.map((slide, index) => {{
    const start = index === 0 ? 0 : currentStart - slide.transitionFrames;
    const duration = slide.durationFrames + (index === 0 ? 0 : slide.transitionFrames);
    currentStart = start + slide.durationFrames;
    return {{
      ...slide,
      sequenceStart: start,
      sequenceDuration: duration,
      entryTransitionDuration: index === 0 ? 0 : slide.transitionFrames
    }};
  }});

  return (
    <AbsoluteFill style={{{{ backgroundColor: '#000000' }}}}>
      {{slideSequences.map((slide, index) => {{
        return (
          <Sequence
            key={{index}}
            from={{slide.sequenceStart}}
            durationInFrames={{slide.sequenceDuration}}
          >
            <SlideComponent slide={{slide}} index={{index}} />
          </Sequence>
        );
      }})}}
    </AbsoluteFill>
  );
}};

const SlideComponent: React.FC<{{ slide: any; index: number }}> = ({{ slide, index }}) => {{
  const frame = useCurrentFrame();
  const {{ fps }} = useVideoConfig();
  
  let opacity = 1;
  let transform = 'scale(1)';
  
  if (slide.entryTransitionDuration > 0 && frame < slide.entryTransitionDuration) {{
    const progress = frame / slide.entryTransitionDuration;
    
    if (slide.transitionType === 'fade') {{
      opacity = interpolate(frame, [0, slide.entryTransitionDuration], [0, 1], {{
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      }});
    }} else if (slide.transitionType === 'zoom') {{
      opacity = interpolate(frame, [0, slide.entryTransitionDuration], [0, 1]);
      const scale = interpolate(frame, [0, slide.entryTransitionDuration], [0.8, 1.0], {{
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      }});
      transform = `scale(${{scale}})`;
    }} else if (slide.transitionType === 'slide') {{
      opacity = interpolate(frame, [0, slide.entryTransitionDuration], [0.3, 1.0]);
      const translateX = interpolate(frame, [0, slide.entryTransitionDuration], [1000, 0], {{
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      }});
      transform = `translateX(${{translateX}}px)`;
    }}
  }}

  const activeFrame = slide.entryTransitionDuration > 0 ? frame - slide.entryTransitionDuration : frame;
  const zoomFactor = interpolate(frame, [0, slide.sequenceDuration], [1.0, 1.08], {{
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  }});
  
  if (slide.transitionType !== 'zoom') {{
    transform = `${{transform}} scale(${{zoomFactor}})`;
  }}

  const captionStartFrame = slide.entryTransitionDuration + 10;
  const captionFrame = frame - captionStartFrame;
  const captionScale = captionFrame > 0 
    ? spring({{ frame: captionFrame, fps, config: {{ damping: 12 }} }}) 
    : 0;

  return (
    <AbsoluteFill style={{{{ overflow: 'hidden' }}}}>
      <div
        style={{{{
          width: '100%',
          height: '100%',
          opacity,
          transform,
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
        }}}}
      >
        <Img
          src={{slide.imagePath}}
          style={{{{
            width: '100%',
            height: '100%',
            objectFit: 'cover',
          }}}}
        />
      </div>

      {{slide.caption && (
        <div
          style={{{{
            position: 'absolute',
            bottom: '15%',
            left: '10%',
            right: '10%',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            textAlign: 'center',
            zIndex: 10,
            transform: `scale(${{captionScale}})`,
            opacity: captionScale,
          }}}}
        >
          <span
            style={{{{
              fontFamily: slide.fontFamily,
              fontSize: slide.fontSize,
              color: slide.fontColor,
              backgroundColor: 'rgba(0, 0, 0, 0.6)',
              padding: '15px 30px',
              borderRadius: '12px',
              fontWeight: 'bold',
              boxShadow: '0 8px 30px rgba(0,0,0,0.3)',
              textShadow: '0 2px 4px rgba(0,0,0,0.5)',
            }}}}
          >
            {{slide.caption}}
          </span>
        </div>
      )}}
    </AbsoluteFill>
  );
}};
"""


# 5. Compiler & Fixer Agent
def agent_compile_and_fix(state: AgentState, project_dir: str) -> Dict[str, Any]:
    remotion_script = state["remotion_script"]
    storyboard = state["storyboard"]
    retry_count = state.get("retry_count", 0)
    errors = state.get("compile_errors", [])
    
    print(f"[Compiler & Fixer] Attempting compilation (Attempt {retry_count + 1})...")
    
    # 1. Setup paths
    app_dir = os.path.join(project_dir, "remotion-app")
    if not os.path.exists(app_dir):
        # Copy template
        template_dir = os.path.join(project_dir, "remotion_template")
        shutil.copytree(template_dir, app_dir)
        print("[Compiler & Fixer] Created remotion-app from template.")
        
    slideshow_path = os.path.join(app_dir, "src", "Slideshow.tsx")
    root_path = os.path.join(app_dir, "src", "Root.tsx")
    
    # 2. Write generated slideshow component
    with open(slideshow_path, "w", encoding="utf-8") as f:
        f.write(remotion_script)
        
    # Calculate duration in frames
    total_frames = 0
    let_start = 0
    for index, slide in enumerate(storyboard.slides):
        duration_frames = int(slide.duration_sec * 30)
        transition_frames = int(slide.transition_duration_sec * 30)
        start = 0 if index == 0 else let_start - transition_frames
        let_start = start + duration_frames
    total_frames = let_start
    if total_frames <= 0:
        total_frames = 300 # Safety default
        
    # 3. Update Root.tsx with correct durationInFrames
    with open(root_path, "r", encoding="utf-8") as f:
        root_content = f.read()
    
    # Simple regex replace of durationInFrames
    updated_root = re.sub(
        r"durationInFrames=\{\d+\}",
        f"durationInFrames={{{total_frames}}}",
        root_content
    )
    with open(root_path, "w", encoding="utf-8") as f:
        f.write(updated_root)

    # 4. Compile check
    node_available = check_node_available()
    if node_available:
        print("[Compiler & Fixer] Node.js is available. Running real compilation...")
        try:
            # Install npm dependencies if node_modules doesn't exist
            node_modules_path = os.path.join(app_dir, "node_modules")
            if not os.path.exists(node_modules_path):
                print("[Compiler & Fixer] Running npm install (this may take a bit)...")
                # We run npm install
                subprocess.run(["npm", "install"], cwd=app_dir, shell=True, check=True, capture_output=True)
            
            # Run Remotion type check / compile test using remotion CLI
            # We run npx remotion render src/index.ts Slideshow --still=0
            # This only compiles and renders the first frame (frame 0), which is extremely fast and tests full compilation!
            print("[Compiler & Fixer] Compiling slideshow frame 0...")
            res = subprocess.run(
                ["npx", "remotion", "render", "src/index.ts", "Slideshow", "out/still.png", "--still=0", "--overwrite"],
                cwd=app_dir,
                shell=True,
                capture_output=True,
                text=True
            )
            
            if res.returncode == 0:
                print("[Compiler & Fixer] Compilation SUCCESS!")
                return {
                    "status": "compiled",
                    "retry_count": retry_count
                }
            else:
                error_msg = res.stderr or res.stdout
                print(f"[Compiler & Fixer] Compilation FAILED: {error_msg[:200]}")
                errors.append(error_msg)
                return {
                    "status": "compile_failed",
                    "compile_errors": errors,
                    "retry_count": retry_count + 1
                }
                
        except Exception as e:
            print(f"[Compiler & Fixer] Compilation command failed to run: {e}")
            errors.append(str(e))
            return {
                "status": "compile_failed",
                "compile_errors": errors,
                "retry_count": retry_count + 1
            }
    else:
        # Mock/Validation Mode
        print("[Compiler & Fixer] Node.js is NOT available. Running syntax validator...")
        # Check code structure
        errors_found = []
        
        # Simple structural syntax checks
        if "import " not in remotion_script:
            errors_found.append("Syntax Error: Missing import statements.")
        if "export const Slideshow" not in remotion_script:
            errors_found.append("Syntax Error: Slideshow component must be exported as 'export const Slideshow'.")
        if "useCurrentFrame" not in remotion_script:
            errors_found.append("Linter Warning: useCurrentFrame is required to animate transitions.")
            
        # Check braces balance
        open_braces = remotion_script.count("{")
        close_braces = remotion_script.count("}")
        if open_braces != close_braces:
            errors_found.append(f"Syntax Error: Unbalanced curly braces. Opened: {open_braces}, Closed: {close_braces}")
            
        # Check parentheses balance
        open_parens = remotion_script.count("(")
        close_parens = remotion_script.count(")")
        if open_parens != close_parens:
            errors_found.append(f"Syntax Error: Unbalanced parentheses. Opened: {open_parens}, Closed: {close_parens}")

        # Check for intentionally injected compilation error for testing (e.g. if script has "INJECT_ERROR")
        if "INJECT_ERROR" in remotion_script:
            errors_found.append("ReferenceError: INJECT_ERROR is not defined.")

        if errors_found:
            error_msg = "\n".join(errors_found)
            print(f"[Compiler & Fixer] Mock compilation FAILED: {error_msg}")
            errors.append(error_msg)
            return {
                "status": "compile_failed",
                "compile_errors": errors,
                "retry_count": retry_count + 1
            }
        else:
            print("[Compiler & Fixer] Mock compilation SUCCESS!")
            return {
                "status": "compiled",
                "retry_count": retry_count
            }


# 6. Renderer Agent
def agent_render(state: AgentState, project_dir: str) -> Dict[str, Any]:
    print("[Renderer] Starting video render...")
    app_dir = os.path.join(project_dir, "remotion-app")
    out_dir = os.path.join(app_dir, "out")
    os.makedirs(out_dir, exist_ok=True)
    
    node_available = check_node_available()
    if node_available:
        print("[Renderer] Running 'npx remotion render' to render video...")
        output_mp4 = os.path.join(out_dir, "video.mp4")
        try:
            res = subprocess.run(
                ["npx", "remotion", "render", "src/index.ts", "Slideshow", "out/video.mp4", "--overwrite"],
                cwd=app_dir,
                shell=True,
                capture_output=True,
                text=True
            )
            if res.returncode == 0:
                print(f"[Renderer] Render success! Output saved to: {output_mp4}")
                return {
                    "status": "success",
                    "output_video_path": output_mp4
                }
            else:
                print(f"[Renderer] Render failed: {res.stderr or res.stdout}")
                return {
                    "status": "failed_rendering",
                    "output_video_path": None
                }
        except Exception as e:
            print(f"[Renderer] Render command error: {e}")
            return {
                "status": "failed_rendering",
                "output_video_path": None
            }
    else:
        print("[Renderer] Node.js is NOT available. Mocking render output...")
        output_mp4 = os.path.join(out_dir, "video_mock.mp4")
        with open(output_mp4, "w", encoding="utf-8") as f:
            f.write("MOCK VIDEO MP4 CONTENT - REMOTION RENDERING SUCCESS")
        print(f"[Renderer] Mock render success! Saved to: {output_mp4}")
        return {
            "status": "success",
            "output_video_path": output_mp4
        }
