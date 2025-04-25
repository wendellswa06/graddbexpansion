import cv2
import os
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
from huggingface_hub import hf_hub_download
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

class RequestData(BaseModel):    
    input_dir: str = Field(default="./")

app = FastAPI()

def get_average_rgb(img):
    # Check if img is a numpy array (OpenCV image) and convert to PIL
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Get all pixels
    pixels = list(img.getdata())
    pixel_count = len(pixels)
    
    # Sum all values
    r_total = sum(pixel[0] for pixel in pixels)
    g_total = sum(pixel[1] for pixel in pixels)
    b_total = sum(pixel[2] for pixel in pixels)
    
    # Calculate averages
    avg_r = r_total / pixel_count
    avg_g = g_total / pixel_count
    avg_b = b_total / pixel_count
    
    return int(avg_r), int(avg_g), int(avg_b)

def detect_and_save_persons(image_path, model_path="yolov8l.pt", conf_threshold=0.5):
    
    # Create output directory
    output_dir = "detected_faces"
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="./model.pt")

    # Load the YOLO model
    model = YOLO(model_path)
    model.conf = conf_threshold
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return 0
        
    height, width = image.shape[:2]
    
    # Convert BGR to RGB (YOLO expects RGB images)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = model(image_rgb)
    
    # Create a copy for drawing all detections
    annotated_image = image_rgb.copy()
    
    # Initialize person counter
    count = 0
    
    # Extract base filename without extension for naming the output files
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Process detections
    for result in results:
        boxes = result.boxes
        
        # Remove this line if you want to process images with multiple faces
        if len(boxes) > 1:
           return {"count": len(boxes), "rgbvalue": (0, 0, 0)}           
        
        for box in boxes:
            count += 1
                          
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            face_image = image[y1:y2, x1:x2]
            
            # Calculate average RGB (if needed)
            avg_r, avg_g, avg_b = get_average_rgb(face_image)
            print(f"Face {count} average RGB: ({avg_r}, {avg_g}, {avg_b})")
            return {"count": 1, "rgbvalue": (avg_r, avg_g, avg_b)}           


@app.post("/fetch_color")
def serve(data: RequestData):
    image_folder_path = data.input_dir
    recursive=False
    if not os.path.isdir(image_folder_path):
        print(f"Error: The folder '{image_folder_path}' does not exist.")
        return
    
    # Count for statistics
    processed_count = 0
    error_count = 0
    
    # Use os.walk to traverse directory and subdirectories
    for root, dirs, files in os.walk(image_folder_path):
        for filename in files:
            print(filename)
            # Get the file extension
            _, extension = os.path.splitext(filename)
            
            # Check if the file is an image (png or jpg)
            if extension.lower() in ['.png', '.jpg', '.jpeg']:
                # Construct the full file path
                file_path = os.path.join(root, filename)
                
                try:
                    # Apply the function to the image
                    result = detect_and_save_persons(file_path)
                    if result.get('count') == 1:
                        return {"success": True, 'rgbvalue': result.get('rgbvalue')}
                    print(f"Processed: {file_path}")
                    processed_count += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    error_count += 1
            # If not recursive, break after first iteration (only process top-level directory)
        if not recursive:
            break
    print(f"Processing complete. Successfully processed {processed_count} images, encountered {error_count} errors.")
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8094)
    # Example usage
    image_path = "5.png"  # Replace with your image path
    # count = detect_and_save_persons(image_path)
    value_r, value_g, value_b = serve('/workspace/portrait-detect')
    print(value_r, value_g, value_b)