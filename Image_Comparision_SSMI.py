import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from datetime import datetime

# Create the folder if it doesn't exist
output_dir = "Image Comparision\Matched Comparison"
os.makedirs(output_dir, exist_ok=True)

def load_image(image_path):
    """Load an image from a file path and convert it to grayscale."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

def resize_image_to_match(image, target_shape):
    """Resize an image to match the dimensions of the target shape."""
    return cv2.resize(image, (target_shape[1], target_shape[0]))  # Width, height order

def calculate_similarity_and_diff(imageA, imageB):
    """Calculate the SSIM similarity score and get the difference mask between two grayscale images."""
    score, diff = ssim(imageA, imageB, full=True)
    diff = (diff * 255).astype("uint8")  # Scale diff image to 0-255 for display
    return score * 100, diff  # Return score as a percentage and the diff mask

def highlight_differences(imageA, imageB, diff):
    """Highlight differences on both images using contours based on the difference mask."""
    # Threshold the diff mask to find regions with differences
    _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours in the thresholded diff mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes around differences in both images
    for contour in contours:
        if cv2.contourArea(contour) > 50:  # Filter out small differences
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for left image
            cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for right image

def create_side_by_side_image(imageA, imageB, imageA_path, imageB_path, match_score):
    """Combine two images side by side, add paths, timestamps, match score, and a separator line."""
    # Concatenate images side by side
    combined_image = np.hstack((imageA, imageB))
    
    # Add a red line in the middle as a separator
    middle_x = imageA.shape[1]
    cv2.line(combined_image, (middle_x, 0), (middle_x, combined_image.shape[0]), (0, 0, 255), 2)

    # Add timestamp at the top of the image
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(combined_image, f"Timestamp: {timestamp}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Add file paths and match score at the bottom
    height, width = combined_image.shape[:2]
    cv2.putText(combined_image, f"{imageA_path}", (10, height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Red color
    cv2.putText(combined_image, f"{imageB_path}", (width // 2 + 10, height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Red color
    cv2.putText(combined_image, f"Match: {match_score:.2f}%", (width // 2 - 100, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # Red color for match score
    
    return combined_image

def compare_and_save(imageA_path, imageB_path):
    """Compare two images and save side-by-side if similarity is above a threshold."""
    imageA, grayA = load_image(imageA_path)
    imageB, grayB = load_image(imageB_path)
    
    # Resize images to match dimensions if they differ
    if grayA.shape != grayB.shape:
        grayB = resize_image_to_match(grayB, grayA.shape)
        imageB = resize_image_to_match(imageB, imageA.shape)
    
    # Calculate similarity and get difference mask
    similarity_score, diff = calculate_similarity_and_diff(grayA, grayB)
    print(f"Similarity between images: {similarity_score:.2f}%")
    
    # Highlight differences on both images
    highlight_differences(imageA, imageB, diff)

    # Save image if similarity meets the threshold
    if similarity_score < 100:  # Save even if images are not identical
        side_by_side_image = create_side_by_side_image(imageA, imageB, imageA_path, imageB_path, similarity_score)
        
        # Save the combined image with timestamp in the filename
        output_path = os.path.join(output_dir, f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        cv2.imwrite(output_path, side_by_side_image)
        print(f"Saved comparison image at: {output_path}")
    else:
        print("Images are identical or do not meet the difference threshold.")

# Example usage with actual paths
compare_and_save(r"Image Comparision\Input Images\Meter1.jpg", 
                 r"Image Comparision\Input Images\PSM meter.jpg")
