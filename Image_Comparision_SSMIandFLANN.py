import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from datetime import datetime

# Create the folder if it doesn't exist
output_dir = "/mnt/data/Matched Comparison"
os.makedirs(output_dir, exist_ok=True)

def load_image(image_path):
    """Load an image from a file path and convert it to grayscale."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

def resize_image_to_match(image, target_shape):
    """Resize an image to match the dimensions of the target shape."""
    return cv2.resize(image, (target_shape[1], target_shape[0]))  # Width, height order

def calculate_combined_similarity(imageA, imageB):
    """Calculate a combined similarity score using SSIM and FLANN-based feature matching."""
    
    # Calculate SSIM similarity
    ssim_score, _ = ssim(imageA, imageB, full=True)
    
    # Set up SIFT detector for FLANN-based matching
    sift = cv2.SIFT_create()
    keypointsA, descriptorsA = sift.detectAndCompute(imageA, None)
    keypointsB, descriptorsB = sift.detectAndCompute(imageB, None)
    
    # FLANN-based matcher setup
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Perform feature matching and filter good matches
    matches = flann.knnMatch(descriptorsA, descriptorsB, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]  # Apply ratio test
    
    # Calculate similarity based on good matches
    flann_similarity_score = (len(good_matches) / max(len(keypointsA), len(keypointsB))) * 100
    
    # Combine SSIM and FLANN similarity scores with a weighted average
    combined_score = (0.5 * (ssim_score * 100)) + (0.5 * flann_similarity_score)
    
    return combined_score, good_matches, keypointsA, keypointsB

def create_side_by_side_image(imageA, imageB, imageA_path, imageB_path, match_score, matches=None, keypointsA=None, keypointsB=None):
    """Combine two images side by side, add paths, timestamps, and match score."""
    # Draw matches if available
    if matches and keypointsA and keypointsB:
        matched_image = cv2.drawMatches(imageA, keypointsA, imageB, keypointsB, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    else:
        # Concatenate images side by side
        matched_image = np.hstack((imageA, imageB))
    
    # Resize matched image for consistency
    height, width = matched_image.shape[:2]
    resized_combined = cv2.resize(matched_image, (width, height))
    
    # Add timestamps
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(resized_combined, f"Timestamp: {timestamp}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Add paths and match score at bottom
    cv2.putText(resized_combined, f"{imageA_path}", (10, height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(resized_combined, f"{imageB_path}", (width // 2 + 10, height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(resized_combined, f"Match: {match_score:.2f}%", (width // 2 - 100, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return resized_combined

def compare_and_save(imageA_path, imageB_path):
    """Compare two images and save side-by-side if similarity is above a threshold."""
    imageA, grayA = load_image(imageA_path)
    imageB, grayB = load_image(imageB_path)
    
    # Resize images to match dimensions if they differ
    if grayA.shape != grayB.shape:
        grayB = resize_image_to_match(grayB, grayA.shape)
        imageB = resize_image_to_match(imageB, imageA.shape)
    
    # Calculate combined similarity
    similarity_score, good_matches, keypointsA, keypointsB = calculate_combined_similarity(grayA, grayB)
    print(f"Combined Similarity between images: {similarity_score:.2f}%")
    
    # Save image if similarity meets the threshold
    if similarity_score > 80:  # Threshold set to 80% similarity
        side_by_side_image = create_side_by_side_image(imageA, imageB, imageA_path, imageB_path, similarity_score, good_matches, keypointsA, keypointsB)
        
        # Save the combined image with timestamp in the filename
        output_path = os.path.join(output_dir, f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        cv2.imwrite(output_path, side_by_side_image)
        print(f"Saved comparison image at: {output_path}")
    else:
        print("Images do not meet the similarity threshold.")

# Paths to the images you uploaded
imageA_path = r"C:\Users\Tspl\Downloads\decision-1013712_1280.webp"
imageB_path = r"C:\Users\Tspl\Downloads\decision-1013751_1280.webp"

# Run the comparison and save the output
compare_and_save(imageA_path, imageB_path)
