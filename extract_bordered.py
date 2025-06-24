from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np
import os

def extract_bordered_slides(pdf_path, output_folder, poppler_path):
    """
    Extracts slides with borders from all pages of a PDF file and saves them in a folder.
    
    :param pdf_path: Path to the PDF file.
    :param output_folder: Folder where the bordered slides will be saved.
    :param poppler_path: Path to the Poppler installation directory.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Convert all PDF pages to images
    slides = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)

    for page_number, slide in enumerate(slides, start=1):
        slide_img = np.array(slide)
        gray = cv2.cvtColor(slide_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        edged = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        best_rect = None
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area > max_area:
                max_area = area
                best_rect = (x, y, w, h)

        if best_rect:
            x, y, w, h = best_rect
            cropped = slide_img[y:y+h, x:x+w]
            output_path = os.path.join(output_folder, f"bordered_slide_page_{page_number}.png")
            Image.fromarray(cropped).save(output_path)
            print(f"Cropped slide saved as '{output_path}' with dimensions: {w}x{h}")
        else:
            print(f"No suitable border found on page {page_number}.")

# Example usage
pdf_path = "Data/Test.pdf"  # Update the path to your PDF file
output_folder = "BorderedSlides"  # Folder to save the bordered slides
poppler_path = r"C:\poppler-24.08.0\Library\bin"  # Update this to the path where Poppler is installed

extract_bordered_slides(pdf_path, output_folder, poppler_path)