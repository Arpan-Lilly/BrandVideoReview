import cv2
import os
import numpy as np

def is_similar_slide(slide1, slide2, threshold=0.9):
    """
    Checks if two slides are similar based on histogram comparison.
    :param slide1: Path to the first slide.
    :param slide2: Path to the second slide.
    :param threshold: Similarity threshold (0.0 to 1.0).
    :return: True if slides are similar, False otherwise.
    """
    img1 = cv2.imread(slide1)
    img2 = cv2.imread(slide2)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute histograms
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

    # Normalize histograms
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()

    # Compute similarity
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity > threshold

def save_unique_slides(input_folder, output_folder, similarity_threshold=0.9):
    """
    Saves only unique slides from the input folder to the output folder.
    :param input_folder: Folder containing all slides.
    :param output_folder: Folder to save unique slides.
    :param similarity_threshold: Threshold for slide similarity (0.0 to 1.0).
    """
    os.makedirs(output_folder, exist_ok=True)

    slides = sorted(os.listdir(input_folder))
    if not slides:
        print("No slides found in the input folder.")
        return

    unique_slide_count = 0
    unique_slide_paths = []  # List to store paths of saved unique slides

    for slide in slides:
        slide_path = os.path.join(input_folder, slide)
        is_unique = True

        # Compare the current slide with all previously saved unique slides
        for unique_slide_path in unique_slide_paths:
            if is_similar_slide(slide_path, unique_slide_path, similarity_threshold):
                is_unique = False
                break

        # Save the slide if it is unique
        if is_unique:
            unique_slide_count += 1
            output_path = os.path.join(output_folder, f"unique_slide_{unique_slide_count}.png")
            cv2.imwrite(output_path, cv2.imread(slide_path))
            unique_slide_paths.append(output_path)  # Add to the list of saved unique slides
            print(f"Saved unique slide {unique_slide_count}: {output_path}")

    print(f"Total unique slides saved: {unique_slide_count}")

if __name__ == "__main__":
    save_unique_slides(input_folder="frames", output_folder="unique_slides", similarity_threshold=0.9)