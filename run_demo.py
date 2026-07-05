import os
import shutil
from generate_mock_images import generate_mock_images
from run_pipeline import run

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(project_dir, "test_images")
    
    # 1. Generate mock images if they don't exist
    if not os.path.exists(images_dir) or len(os.listdir(images_dir)) == 0:
        print("Test images directory not found. Generating mock event photos...")
        generate_mock_images(images_dir, num_images=10)
    else:
        print(f"Using existing mock images in: {images_dir}")
        
    # 2. Run Demo 1: Cinematic Wedding
    wedding_prompt = "Cinematic wedding reel, slow and emotional, warm tones, minimal text"
    print("\n" + "=" * 80)
    print("RUNNING DEMO 1: Cinematic Wedding Reel")
    print("=" * 80)
    run(images_dir, wedding_prompt, max_retries=3, output_dir="sample_output/wedding")
    
    # 3. Run Demo 2: Upbeat Birthday
    birthday_prompt = "Upbeat birthday reel, fast cuts, bold captions, energetic"
    print("\n" + "=" * 80)
    print("RUNNING DEMO 2: Upbeat Birthday Reel")
    print("=" * 80)
    run(images_dir, birthday_prompt, max_retries=3, output_dir="sample_output/birthday")
    
    print("\n" + "=" * 80)
    print("Demo Execution Complete!")
    print("Check the 'sample_output/wedding' and 'sample_output/birthday' directories for:")
    print("  1. storyboard.json (sequences, pacing, colors, captions)")
    print("  2. Slideshow.tsx (the compiled Remotion slideshow component)")
    print("  3. pipeline_state.json (full trace of the agent state)")
    print("=" * 80)

if __name__ == "__main__":
    main()
