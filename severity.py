import cv2
import numpy as np
import os

def calculate_disease_severity(input_image_path, output_image_name=None):
    # Load the image
    image = cv2.imread(input_image_path)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper HSV color range for diseased areas
    lower_color = np.array([0, 50, 50])
    upper_color = np.array([20, 255, 255])

    # Create a mask for the diseased areas
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Apply morphological operations to enhance the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables for total area and affected area
    total_area = image.shape[0] * image.shape[1]  # Total area of the image
    affected_area = 0  # Accumulated area affected by disease

    # Draw bounding boxes around the contours and calculate affected area
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        affected_area += w * h

    # Calculate the severity percentage
    severity_percentage = round(float((affected_area / total_area) * 100),3)

    # Print the severity percentage and output image path
    if output_image_name is None:
        output_image_name = f"output_{os.path.basename(input_image_path)}"
    output_image_path = os.path.join('static/upload/', output_image_name)
    cv2.imwrite(output_image_path, image)
    return severity_percentage, output_image_path