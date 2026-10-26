"""
Create a slide montage of all 18 slides to inspect the overall rhythm and layout.
"""
import os
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SLIDES_DIR = os.path.join(BASE_DIR, "presentation_slides")
MONTAGE_PATH = os.path.join(BASE_DIR, "presentation_slides", "montage_all_slides.png")

cols = 3
rows = 6
slide_w = 480
slide_h = 270

montage = Image.new("RGB", (cols * slide_w, rows * slide_h), (20, 20, 20))

for idx in range(1, 19):
    img_path = os.path.join(SLIDES_DIR, f"slide_{idx:02d}.png")
    if os.path.exists(img_path):
        im = Image.open(img_path)
        im = im.resize((slide_w, slide_h), Image.Resampling.LANCZOS)
        col = (idx - 1) % cols
        row = (idx - 1) // cols
        montage.paste(im, (col * slide_w, row * slide_h))

montage.save(MONTAGE_PATH)
print(f"Montage saved successfully to {MONTAGE_PATH} (size: {montage.size})")
