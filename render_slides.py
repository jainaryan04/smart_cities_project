"""
Render all slides of final_presentation.pptx to PNG images using PowerPoint COM.
"""
import os
import win32com.client

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PPTX_PATH = os.path.join(BASE_DIR, "final_presentation.pptx")
OUTPUT_DIR = os.path.join(BASE_DIR, "presentation_slides")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Opening PowerPoint to export slides from {PPTX_PATH}...")
ppt = win32com.client.Dispatch("PowerPoint.Application")
try:
    pres = ppt.Presentations.Open(os.path.abspath(PPTX_PATH), WithWindow=False)
    num_slides = pres.Slides.Count
    print(f"Total slides found: {num_slides}")
    for i in range(1, num_slides + 1):
        slide = pres.Slides(i)
        out_file = os.path.join(OUTPUT_DIR, f"slide_{i:02d}.png")
        slide.Export(os.path.abspath(out_file), "PNG", 1920, 1080)
        print(f"Exported Slide {i:02d} -> {out_file}")
    pres.Close()
finally:
    ppt.Quit()
print("All slides exported successfully to presentation_slides directory!")
