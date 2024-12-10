import os
import cv2
import numpy as np
import pytesseract  # for OCR
from skimage.metrics import structural_similarity as ssim
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Create folders if they don't exist
output_dir = "Image Comparison/Matched Comparison"
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

def ocr_text_detection(image):
    """Detect text in an image using OCR."""
    config = "--psm 6"  # Treat the image as a single uniform block of text
    text = pytesseract.image_to_string(image, config=config)
    return text.strip()

def compare_texts(textA, textB):
    """Compare detected texts and report differences."""
    if textA != textB:
        return f"Text differs: '{textA}' vs '{textB}'"
    return None

def calculate_color_difference(imageA, imageB):
    """Calculate the average color difference between two images."""
    mean_colorA = cv2.mean(imageA)[:3]  # Average BGR color values
    mean_colorB = cv2.mean(imageB)[:3]
    color_diff = np.sqrt(np.sum((np.array(mean_colorA) - np.array(mean_colorB)) ** 2))
    return color_diff

def describe_color_difference(color_diff):
    """Generate a description based on color difference."""
    if color_diff > 50:  # Threshold for notable color difference
        return f"Significant color difference detected (difference score: {color_diff:.2f})."
    return None

def highlight_differences(imageA, imageB, diff):
    """Highlight differences on both images using contours based on the difference mask."""
    _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    diff_count = 0
    diff_details = []
    for contour in contours:
        if cv2.contourArea(contour) > 50:  # Filter out small differences
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for left image
            cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for right image
            diff_count += 1
            diff_details.append(f"Difference {diff_count}: Region (x={x}, y={y}, w={w}, h={h})")
    return diff_count, diff_details

def create_side_by_side_image(imageA, imageB, imageA_path, imageB_path, match_score):
    """Combine two images side by side with labels, timestamps, and a separator line."""
    combined_image = np.hstack((imageA, imageB))
    middle_x = imageA.shape[1]
    cv2.line(combined_image, (middle_x, 0), (middle_x, combined_image.shape[0]), (0, 0, 255), 2)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(combined_image, f"Timestamp: {timestamp}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    height, width = combined_image.shape[:2]
    cv2.putText(combined_image, f"{imageA_path}", (10, height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(combined_image, f"{imageB_path}", (width // 2 + 10, height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(combined_image, f"Match: {match_score:.2f}%", (width // 2 - 100, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return combined_image

def generate_report(image_path, report_path, similarity_score, imageA_path, imageB_path, diff_count, diff_details, additional_notes):
    """Generate a detailed PDF report with conclusions on image similarity."""
    c = canvas.Canvas(report_path, pagesize=letter)
    c.drawString(30, 750, "Image Comparison Report")
    c.drawString(30, 730, f"Similarity Score: {similarity_score:.2f}%")
    c.drawString(30, 710, f"Image 1 Path: {imageA_path}")
    c.drawString(30, 690, f"Image 2 Path: {imageB_path}")
    c.drawString(30, 670, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Add a conclusion based on the similarity score
    if similarity_score >= 90:
        conclusion = "Images are almost identical with minimal differences."
    elif similarity_score >= 70:
        conclusion = f"Images have some differences, with {diff_count} unique attributes found."
    else:
        conclusion = f"Images differ significantly, with {diff_count} different attributes highlighted."
    c.drawString(30, 650, f"Conclusion: {conclusion}")
    
    # Add detailed differences
    y_position = 630
    c.drawString(30, y_position, "Detailed Differences:")
    for detail in diff_details:
        y_position -= 15
        c.drawString(30, y_position, detail)
        if y_position < 150:  # Adjust position if reaching the bottom of the page
            c.showPage()
            y_position = 750
    
    # Add additional notes
    y_position -= 20
    c.drawString(30, y_position, "Additional Analysis:")
    for note in additional_notes:
        y_position -= 15
        c.drawString(30, y_position, note)
        if y_position < 150:
            c.showPage()
            y_position = 750
    
    # Insert the comparison image
    comparison_image = ImageReader(image_path)
    c.drawImage(comparison_image, 30, y_position - 300, width=540, height=300)
    
    c.showPage()
    c.save()
    print(f"Report saved at: {report_path}")

def compare_and_save(imageA_path, imageB_path):
    """Compare two images, save side-by-side comparison, and generate a PDF report."""
    imageA, grayA = load_image(imageA_path)
    imageB, grayB = load_image(imageB_path)
    
    if grayA.shape != grayB.shape:
        grayB = resize_image_to_match(grayB, grayA.shape)
        imageB = resize_image_to_match(imageB, imageA.shape)
    
    similarity_score, diff = calculate_similarity_and_diff(grayA, grayB)
    print(f"Similarity between images: {similarity_score:.2f}%")
    
    diff_count, diff_details = highlight_differences(imageA, imageB, diff)
    
    # Additional human-readable analysis
    additional_notes = []
    
    # Text comparison
    textA = ocr_text_detection(grayA)
    textB = ocr_text_detection(grayB)
    text_diff = compare_texts(textA, textB)
    if text_diff:
        additional_notes.append(text_diff)
    
    # Color comparison
    color_diff = calculate_color_difference(imageA, imageB)
    color_diff_note = describe_color_difference(color_diff)
    if color_diff_note:
        additional_notes.append(color_diff_note)
    
    side_by_side_image = create_side_by_side_image(imageA, imageB, imageA_path, imageB_path, similarity_score)
    comparison_image_path = os.path.join(output_dir, f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    cv2.imwrite(comparison_image_path, side_by_side_image)
    print(f"Saved comparison image at: {comparison_image_path}")
    
    report_path = os.path.join(output_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    generate_report(comparison_image_path, report_path, similarity_score, imageA_path, imageB_path, diff_count, diff_details, additional_notes)

# Example usage with actual paths
compare_and_save("Image Comparision\Input Images\Meter1.jpg", 
                 "Image Comparision\Input Images\Meter2.jpg")

 