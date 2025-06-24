from skimage.metrics import structural_similarity as ssim
import cv2
import os
import numpy as np

def is_similar_slide(slide1, slide2, threshold=0.8):
    """
    Checks if two slides are similar based on Structural Similarity Index (SSIM).
    :param slide1: Current slide (numpy array).
    :param slide2: Last saved unique slide (numpy array).
    :param threshold: Similarity threshold (0.0 to 1.0).
    :return: True if slides are similar, False otherwise.
    """
    gray1 = cv2.cvtColor(slide1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(slide2, cv2.COLOR_BGR2GRAY)

    score, _ = ssim(gray1, gray2, full=True)
    print(f"SSIM score: {score}")  # Debugging line
    return score > threshold

def process_video_and_save_unique_slides(video_path, output_folder, pixel_threshold=5000, blur_threshold=150, similarity_threshold=0.8, frame_skip=10):
    """
    Processes a video to extract valid and unique slides, saving them directly to a folder.
    :param video_path: Path to the video file.
    :param output_folder: Folder to save unique slides.
    :param pixel_threshold: Threshold for pixel change detection.
    :param blur_threshold: Threshold for blur detection.
    :param similarity_threshold: Threshold for slide similarity (0.0 to 1.0).
    :param frame_skip: Number of frames to skip between processing.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    os.makedirs(output_folder, exist_ok=True)

    ret, prev_frame = cap.read()
    frame_no = 0
    valid_slide_count = 0
    last_unique_slide = None  # Store the last saved unique slide

    while ret:
        ret, frame = cap.read()
        frame_no += 1

        # Skip frames based on the frame_skip interval
        if frame_no % frame_skip != 0:
            continue

        if not ret:
            break

        # Print the current frame being checked
        print(f"Checking frame {frame_no}...")

        if prev_frame is not None:
            # Process every frame
            transition = is_transition_frame(prev_frame, frame, pixel_threshold)
            blurred = is_blurred_frame(frame, blur_threshold)

            if not transition and not blurred:
                # Compare only with the last saved unique slide
                is_unique = True
                if last_unique_slide is not None and is_similar_slide(frame, last_unique_slide, similarity_threshold):
                    is_unique = False

                # Save only if unique
                if is_unique:
                    valid_slide_count += 1
                    output_path = os.path.join(output_folder, f"unique_slide_{valid_slide_count}.png")
                    cv2.imwrite(output_path, frame)
                    last_unique_slide = frame  # Update the last saved unique slide
                    print(f"Saved unique slide {valid_slide_count}: {output_path}")

        prev_frame = frame

    cap.release()
    print(f"Total unique slides saved: {valid_slide_count}")

def is_transition_frame(frame1, frame2, threshold=5000):
    """
    Detects if a frame is a transition frame based on pixel changes.
    :param frame1: Previous frame.
    :param frame2: Current frame.
    :param threshold: Pixel change threshold.
    :return: True if the frame is a transition frame, False otherwise.
    """
    diff = cv2.absdiff(frame1, frame2)
    diff_sum = np.sum(diff) / (frame1.shape[0] * frame1.shape[1])  # Normalize by frame size
    return diff_sum > threshold

def is_blurred_frame(frame, threshold=150):
    """
    Detects if a frame is blurred based on Laplacian variance.
    :param frame: Current frame.
    :param threshold: Blur detection threshold.
    :return: True if the frame is blurred, False otherwise.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

if __name__ == "__main__":
    process_video_and_save_unique_slides(
        video_path="Testvid.mp4",
        output_folder="unique_slides",
        pixel_threshold=5000,
        blur_threshold=100,
        similarity_threshold=0.9,  # Adjusted threshold for SSIM
        frame_skip=5  # Skip every 10 frames
    )