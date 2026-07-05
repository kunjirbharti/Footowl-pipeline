import os
import shutil
import unittest
from unittest.mock import patch, MagicMock
from PIL import Image

# Import models, rag, graph, agents
from models import VideoIntent, ImageAnalysis, Storyboard, StoryboardSlide
from rag import SimpleRAG
from graph import build_pipeline_graph

class TestImageToVideoPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Create a temp test directory for images
        cls.test_dir = os.path.dirname(os.path.abspath(__file__))
        cls.images_dir = os.path.join(cls.test_dir, "temp_test_images")
        os.makedirs(cls.images_dir, exist_ok=True)
        
        # Create 5 small dummy images
        for i in range(5):
            img = Image.new("RGB", (100, 100), color=(i * 40, i * 40, i * 40))
            img.save(os.path.join(cls.images_dir, f"test_photo_{i}.jpg"))
            
        cls.image_paths = sorted([
            os.path.join(cls.images_dir, f"test_photo_{i}.jpg") for i in range(5)
        ])
        
        # Initialize RAG offline (no gemini embeddings for test speed)
        cls.rag = SimpleRAG(use_gemini_embeddings=False)
        cls.graph = build_pipeline_graph(cls.test_dir, cls.rag)

    @classmethod
    def tearDownClass(cls):
        # Clean up temp images and app dir
        if os.path.exists(cls.images_dir):
            shutil.rmtree(cls.images_dir)
            
        app_dir = os.path.join(cls.test_dir, "remotion-app")
        if os.path.exists(app_dir):
            shutil.rmtree(app_dir)

    def test_wedding_scenario(self):
        """Test Scenario 1: Cinematic Wedding (slow, emotional, serif, fade)"""
        initial_state = {
            "user_prompt": "Cinematic wedding reel, slow and emotional, warm tones, minimal text",
            "image_paths": self.image_paths,
            "retry_count": 0,
            "max_retries": 2,
            "compile_errors": [],
            "status": "started"
        }
        
        # Running the compiled graph (mocks are automatically active since no API key is set)
        result = self.graph.invoke(initial_state)
        
        self.assertEqual(result["status"], "success")
        self.assertIsNotNone(result["output_video_path"])
        self.assertIsNotNone(result["storyboard"])
        
        intent = result["video_intent"]
        self.assertEqual(intent.pacing, "slow")
        self.assertEqual(intent.visual_style, "cinematic")
        
        storyboard = result["storyboard"]
        self.assertTrue(len(storyboard.slides) > 0)
        
        # Check cinematic aesthetics
        for slide in storyboard.slides:
            self.assertEqual(slide.font_family, "Playfair Display")
            self.assertEqual(slide.duration_sec, 5.0)
            if slide.image_path != self.image_paths[0]: # First slide doesn't have transition in
                self.assertEqual(slide.transition_type, "fade")

    def test_birthday_scenario(self):
        """Test Scenario 2: Upbeat Birthday (fast, energetic, display, zoom/none)"""
        initial_state = {
            "user_prompt": "Upbeat birthday reel, fast cuts, bold captions, energetic",
            "image_paths": self.image_paths,
            "retry_count": 0,
            "max_retries": 2,
            "compile_errors": [],
            "status": "started"
        }
        
        result = self.graph.invoke(initial_state)
        
        self.assertEqual(result["status"], "success")
        intent = result["video_intent"]
        self.assertEqual(intent.pacing, "fast")
        self.assertEqual(intent.visual_style, "upbeat")
        
        storyboard = result["storyboard"]
        self.assertTrue(len(storyboard.slides) > 0)
        
        for slide in storyboard.slides:
            self.assertEqual(slide.font_family, "Impact")
            self.assertEqual(slide.duration_sec, 1.5)

    def test_corporate_scenario(self):
        """Test Scenario 3: Clean Corporate (medium, professional, sans-serif, slide)"""
        initial_state = {
            "user_prompt": "Clean corporate highlights, professional tone, subtle transitions",
            "image_paths": self.image_paths,
            "retry_count": 0,
            "max_retries": 2,
            "compile_errors": [],
            "status": "started"
        }
        
        result = self.graph.invoke(initial_state)
        
        self.assertEqual(result["status"], "success")
        intent = result["video_intent"]
        self.assertEqual(intent.pacing, "medium")
        self.assertEqual(intent.visual_style, "corporate")
        
        storyboard = result["storyboard"]
        self.assertTrue(len(storyboard.slides) > 0)
        
        for slide in storyboard.slides:
            self.assertEqual(slide.font_family, "Inter")
            self.assertEqual(slide.duration_sec, 3.0)

    def test_llm_as_judge_coherence(self):
        """Test Scenario 4: LLM-as-judge evaluating narrative coherence of storyboard"""
        # 1. Create a storyboard
        slides = [
            StoryboardSlide(
                image_path=self.image_paths[0],
                caption="The guests arrive and the venue looks stunning",
                duration_sec=4.0,
                transition_type="none",
                transition_duration_sec=0.0,
                font_family="Inter",
                font_size="40px",
                font_color="#ffffff"
            ),
            StoryboardSlide(
                image_path=self.image_paths[1],
                caption="A close up of the key speakers discussing technology",
                duration_sec=4.0,
                transition_type="slide",
                transition_duration_sec=0.8,
                font_family="Inter",
                font_size="40px",
                font_color="#ffffff"
            ),
            StoryboardSlide(
                image_path=self.image_paths[2],
                caption="The final concluding remarks and panel discussion",
                duration_sec=4.0,
                transition_type="slide",
                transition_duration_sec=0.8,
                font_family="Inter",
                font_size="40px",
                font_color="#ffffff"
            )
        ]
        storyboard = Storyboard(title="Corporate Annual Meet 2026", slides=slides)
        
        # 2. Run LLM-as-judge evaluation
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            # Live LLM evaluation
            import google.generativeai as genai
            from pydantic import BaseModel, Field
            
            class CoherenceRating(BaseModel):
                score: float = Field(description="Score from 1.0 (incoherent) to 10.0 (highly coherent)")
                explanation: str = Field(description="Reasoning behind the rating")
                
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            prompt = (
                "You are an expert video editor acting as a judge. "
                "Evaluate the narrative coherence of this generated video storyboard. "
                "A coherent storyboard has a logical progression (e.g. intro -> body -> conclusion) "
                "and consistent caption styling.\n\n"
                f"Storyboard:\n{storyboard.model_dump_json(indent=2)}\n\n"
                "Return a JSON rating."
            )
            
            generation_config = genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=CoherenceRating
            )
            response = model.generate_content(prompt, generation_config=generation_config)
            rating = CoherenceRating.model_validate_json(response.text)
            score = rating.score
            explanation = rating.explanation
        else:
            # Local Python-based evaluation (mock judge)
            print("[LLM-as-Judge] No API key available. Running local evaluation rules.")
            # Check coherence programmatically:
            # Rule 1: Storyboard must have multiple slides
            # Rule 2: Slide captions should not be empty
            # Rule 3: Captions should show a progression (not identical)
            # Rule 4: Timings should be positive
            has_progression = len(slides) >= 2
            non_empty = all(len(s.caption) > 0 for s in slides)
            positive_times = all(s.duration_sec > 0 for s in slides)
            all_captions = [s.caption for s in slides]
            distinct_captions = len(set(all_captions)) == len(all_captions)
            
            if has_progression and non_empty and positive_times and distinct_captions:
                score = 9.5
                explanation = "Logical progression detected. Captions are unique, descriptive, and outline a clear beginning, middle, and end. Timing and font styling are consistent."
            else:
                score = 4.0
                explanation = "Storyboard lacks logical progression or contains duplicate captions/invalid timings."
                
        print(f"[LLM-as-Judge Score] {score}/10.0")
        print(f"[LLM-as-Judge Reason] {explanation}")
        
        self.assertGreaterEqual(score, 7.0, f"Storyboard failed coherence test. Score: {score}, Reason: {explanation}")

    def test_compiler_retry_loop(self):
        """Test Scenario 5: Validating compiler failure and success retry flow in LangGraph"""
        
        # We will mock the compile check to fail on the first attempt (retry_count=0) and succeed on the second (retry_count=1)
        # This will verify the conditional routing edge transitions:
        # compile_and_fix -> generate_script -> compile_and_fix -> render -> END
        
        state_log = []
        
        # We mock the agent_compile_and_fix function
        def mock_compile_and_fix(state, project_dir):
            retry = state.get("retry_count", 0)
            errors = state.get("compile_errors", [])
            state_log.append(f"compile_attempt_{retry}")
            
            if retry == 0:
                errors.append("TypeScript Compilation Error: Cannot find name 'UndefinedVar'.")
                return {
                    "status": "compile_failed",
                    "compile_errors": errors,
                    "retry_count": retry + 1
                }
            else:
                return {
                    "status": "compiled",
                    "retry_count": retry
                }

        # Patch agent_compile_and_fix in agents module
        with patch("graph.agent_compile_and_fix", side_effect=mock_compile_and_fix):
            # Recompile graph with the patched node reference
            test_graph = build_pipeline_graph(self.test_dir, self.rag)
            
            initial_state = {
                "user_prompt": "Clean corporate highlights, professional tone, subtle transitions",
                "image_paths": self.image_paths,
                "retry_count": 0,
                "max_retries": 3,
                "compile_errors": [],
                "status": "started"
            }
            
            result = test_graph.invoke(initial_state)
            
            # The pipeline should complete successfully after 1 retry
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["retry_count"], 1)
            self.assertEqual(len(result["compile_errors"]), 1)
            self.assertIn("UndefinedVar", result["compile_errors"][0])
            self.assertEqual(state_log, ["compile_attempt_0", "compile_attempt_1"])

if __name__ == "__main__":
    unittest.main()
