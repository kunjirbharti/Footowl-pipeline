import os
from PIL import Image, ImageDraw, ImageFont

def generate_mock_images(output_dir="test_images", num_images=10):
    """Generates a folder of solid-color mock images with text labels for testing."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 10 harmonious pastel colors
    colors = [
        (255, 179, 186), # Pink
        (255, 223, 186), # Peach
        (255, 255, 186), # Yellow
        (186, 255, 201), # Mint Green
        (186, 225, 255), # Sky Blue
        (221, 186, 255), # Lavender
        (255, 200, 220), # Rose
        (200, 255, 220), # Seafoam
        (240, 230, 140), # Khaki
        (224, 255, 255)  # Cyan
    ]
    
    print(f"Generating {num_images} mock images in: {output_dir}")
    
    for i in range(num_images):
        color = colors[i % len(colors)]
        # Create an 800x800 RGB image
        img = Image.new("RGB", (800, 800), color=color)
        draw = ImageDraw.Draw(img)
        
        # Draw a simple geometric design (a border and center rectangle)
        draw.rectangle([20, 20, 780, 780], outline=(255, 255, 255), width=8)
        draw.ellipse([300, 300, 500, 500], fill=(255, 255, 255))
        
        # Write text label
        text = f"Event Photo {i+1}"
        # We draw text without external font file (using default bitmap font)
        # To make it readable, we draw a shadow first
        draw.text((365, 395), text, fill=(50, 50, 50))
        draw.text((364, 394), text, fill=(0, 0, 0))
        
        # Save image
        img_path = os.path.join(output_dir, f"photo_{i+1:02d}.jpg")
        img.save(img_path, "JPEG")
        print(f"  Created: {img_path}")
        
    print(f"Mock image generation complete! {num_images} images saved.")

if __name__ == "__main__":
    generate_mock_images()
