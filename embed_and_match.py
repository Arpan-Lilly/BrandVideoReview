from sklearn.metrics.pairwise import cosine_similarity
from clipmodel import get_image_embedding
import os
import numpy as np

def compare_frames_to_slides(slide_folders="slides", frame_folders="frames", threshold=0.85):
    slide_embeddings = []
    slide_names = []
    
    # Process slide embeddings
    for fname in sorted(os.listdir(slide_folders)):
        if fname.endswith('.png'):
            slide_path = os.path.join(slide_folders, fname)
            slide_embedding = get_image_embedding(slide_path)
            slide_embeddings.append(slide_embedding.cpu().numpy().flatten())  # Flatten the embedding
            slide_names.append(fname)

    slide_embeddings = np.array(slide_embeddings)  # Convert to a 2D array

    # Process frame embeddings and compare
    for fname in sorted(os.listdir(frame_folders)):
        if fname.endswith('.png'):
            frame_path = os.path.join(frame_folders, fname)
            frame_embedding = get_image_embedding(frame_path).cpu().numpy().flatten()  # Flatten the embedding
            
            # Compute cosine similarity
            similarities = cosine_similarity(frame_embedding.reshape(1, -1), slide_embeddings)
            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[0, best_match_idx]
            
            if best_similarity < threshold:
                print(f"Frame '{fname}' does not match any slide above the threshold. But max = '{slide_names[best_match_idx]}' with similarity {best_similarity:.2f}")
            
                

if __name__ == "__main__":
    compare_frames_to_slides(slide_folders="slides", frame_folders="unique_slides", threshold=0.80)