#!/usr/bin/env python3
"""
Create placeholder images for the enhanced UI.

This script generates basic placeholder images for the splash screen
and app icon to prevent missing resource warnings.
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def create_splash_image(output_path, width=600, height=400):
    """Create a simple splash screen image"""
    # Create a new image with white background
    image = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    # Draw background with a nice gradient
    for y in range(height):
        # Create a light blue to dark blue gradient
        r = int(30 + (y / height) * (0 - 30))
        g = int(58 + (y / height) * (42 - 58))
        b = int(138 + (y / height) * (96 - 138))
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    # Add text
    title_text = "AI Code Review Tool"
    subtitle_text = "Enhanced UI"
    
    # Use a default font if custom font not available
    title_font_size = 48
    subtitle_font_size = 30
    
    try:
        # Try to use a system font
        title_font = ImageFont.truetype("Arial", title_font_size)
        subtitle_font = ImageFont.truetype("Arial", subtitle_font_size)
    except IOError:
        # Fall back to default font
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
    
    # Position text
    title_width = draw.textlength(title_text, font=title_font)
    title_position = ((width - title_width) // 2, height // 2 - 40)
    
    subtitle_width = draw.textlength(subtitle_text, font=subtitle_font)
    subtitle_position = ((width - subtitle_width) // 2, height // 2 + 20)
    
    # Draw text with a subtle shadow
    # Shadow
    draw.text((title_position[0] + 2, title_position[1] + 2), title_text, fill=(30, 30, 30, 128), font=title_font)
    # Main text
    draw.text(title_position, title_text, fill=(255, 255, 255), font=title_font)
    
    # Subtitle
    draw.text((subtitle_position[0] + 1, subtitle_position[1] + 1), subtitle_text, fill=(30, 30, 30, 128), font=subtitle_font)
    draw.text(subtitle_position, subtitle_text, fill=(220, 220, 220), font=subtitle_font)
    
    # Save the image
    image.save(output_path)
    print(f"Splash image created at: {output_path}")

def create_icon_image(output_path, size=128):
    """Create a simple app icon"""
    # Create a square image with a blue background
    image = Image.new('RGB', (size, size), color=(30, 58, 138))
    draw = ImageDraw.Draw(image)
    
    # Draw a simple code review icon (abstract code lines)
    line_color = (255, 255, 255)
    accent_color = (13, 148, 136)  # Teal color from the design system
    
    # Code lines
    margin = size // 8
    line_height = size // 12
    line_spacing = size // 8
    
    for i in range(4):
        y_pos = margin + i * line_spacing
        # Variable width lines to simulate code
        line_width = int((0.5 + (i % 3) * 0.15) * (size - 2 * margin))
        draw.rectangle([(margin, y_pos), (margin + line_width, y_pos + line_height)], 
                      fill=line_color if i != 2 else accent_color)
    
    # Add a checkmark in the bottom right
    check_size = size // 3
    check_pos = (size - margin - check_size, size - margin - check_size)
    
    # Draw circular background for checkmark
    draw.ellipse([check_pos, (check_pos[0] + check_size, check_pos[1] + check_size)], 
                fill=accent_color)
    
    # Draw checkmark
    check_margin = check_size // 4
    check_points = [
        (check_pos[0] + check_margin, check_pos[1] + check_size // 2),
        (check_pos[0] + check_size // 2.5, check_pos[1] + check_size - check_margin),
        (check_pos[0] + check_size - check_margin, check_pos[1] + check_margin)
    ]
    draw.line(check_points, fill=(255, 255, 255), width=check_size // 6)
    
    # Save the image
    image.save(output_path)
    print(f"Icon image created at: {output_path}")

def main():
    # Get the base directory (where this script is located)
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    base_dir = script_dir.parent if script_dir.name == "assets" else script_dir
    
    # Create assets directory if it doesn't exist
    assets_dir = base_dir / "assets"
    assets_dir.mkdir(exist_ok=True)
    
    # Create splash image
    splash_path = assets_dir / "splash.png"
    create_splash_image(splash_path)
    
    # Create icon image
    icon_path = assets_dir / "icon.png"
    create_icon_image(icon_path)
    
    print("Done! Created placeholder images for the enhanced UI.")

if __name__ == "__main__":
    try:
        from PIL import Image, ImageDraw, ImageFont
        main()
    except ImportError:
        print("Error: PIL (Pillow) library is required. Install it with 'pip install pillow'")
        sys.exit(1)
