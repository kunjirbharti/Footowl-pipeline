from typing import List, Optional, TypedDict
from pydantic import BaseModel, Field

class VideoIntent(BaseModel):
    pacing: str = Field(description="The pacing of the video, e.g., 'slow', 'medium', 'fast'")
    visual_style: str = Field(description="The primary visual style or theme, e.g., 'cinematic', 'upbeat', 'corporate'")
    caption_tone: str = Field(description="The tone of the text overlays, e.g., 'emotional', 'bold', 'professional', 'minimal'")
    transition_preference: str = Field(description="Preferred transition type, e.g., 'fade', 'slide', 'zoom', 'none'")
    font_style: str = Field(description="Recommended font style, e.g., 'serif', 'sans-serif', 'monospace', 'bold-display'")

class ImageAnalysis(BaseModel):
    image_path: str = Field(description="The absolute path of the image")
    subject: str = Field(description="A detailed description of the main subject/action in the image")
    color_palette: List[str] = Field(description="Dominant colors or tones, e.g., ['warm gold', 'brown', 'cream']")
    mood: str = Field(description="The emotional vibe, e.g., 'romantic', 'professional', 'joyful', 'nostalgic'")
    quality_score: float = Field(description="A score from 1.0 to 10.0 for composition and clarity")
    recommended_caption: str = Field(description="A contextually appropriate caption based on the image subject")

class StoryboardSlide(BaseModel):
    image_path: str = Field(description="Path of the image used in this slide")
    caption: str = Field(description="The text overlay displayed on the screen")
    duration_sec: float = Field(description="Duration of this slide in seconds")
    transition_type: str = Field(description="Transition into this slide: 'fade', 'slide', 'zoom', 'none'")
    transition_duration_sec: float = Field(description="Transition duration in seconds")
    font_family: str = Field(description="Font family for the text, e.g., 'Playfair Display', 'Inter', 'Impact'")
    font_size: str = Field(description="Font size with unit, e.g., '48px', '64px'")
    font_color: str = Field(description="Hex color code, e.g., '#ffffff', '#ffcc00'")

class Storyboard(BaseModel):
    title: str = Field(description="The title of the video reel")
    slides: List[StoryboardSlide] = Field(description="The sequential list of slides forming the narrative")

class AgentState(TypedDict, total=False):
    user_prompt: str
    image_paths: List[str]
    video_intent: VideoIntent
    image_analyses: List[ImageAnalysis]
    selected_images: List[str]
    storyboard: Storyboard
    remotion_script: str
    compile_errors: List[str]
    retry_count: int
    max_retries: int
    status: str
    output_video_path: Optional[str]
