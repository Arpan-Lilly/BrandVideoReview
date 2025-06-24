import cv2
import os
def extract_keyframes(video_path, output_folder, threshold=5):
    """
    Extracts keyframes from a video file at specified intervals and saves them in a folder.
    
    :param video_path: Path to the video file.
    :param output_folder: Folder where the keyframes will be saved.
    :param interval: Interval in seconds between keyframes.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    last_gray = None
    frame_no = 0

    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if last_gray is not None:
            diff = cv2.absdiff(gray,last_gray)
            if diff.mean() > threshold:
                # Save the frame as an image
                output_path = os.path.join(output_folder, f"keyframe_{saved_count:04d}.png")
                cv2.imwrite(output_path, frame)
                print(f"Saved keyframe: {output_path}")
                saved_count += 1
        
        last_gray = gray
        frame_no += 1
    cap.release()



if __name__ == "__main__":
    extract_keyframes("Testvid.mp4", output_folder="frames")
