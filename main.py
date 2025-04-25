import requests
import json
import os
import shutil  # Added missing import
from PIL import Image
from io import BytesIO

endpoint_url  = {
    "image_endpoint_url": "http://0.0.0.0:8093/generate",
    "color_endpoint_url": "http://0.0.0.0:8094/fetch_color"    
}


def generate_image(id, prompt, is_person, skin_color, image_style, output_dir):
    os.makedirs(output_dir, exist_ok=True)
       
    # Prepare the request payload
    payload = {
        "id": 1,
        "is_person": is_person,
        "prompt": prompt,
        "output_dir": output_dir,
        "skin_color": skin_color,
        "image_style": image_style,
    }
    
    try:
        # Send the POST request to the endpoint
        response = requests.post(endpoint_url.get('image_endpoint_url'), json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            if result["success"] == True:  # Fixed dictionary access                            
                return {
                    "success": True,
                    "message": "Image generated successfully",
                }
            return {
                    "success": False,
                    "message": "Image generation Failed",
                }
        else:
            # Handle error responses
            return {
                "success": False,
                "message": f"Error: {response.status_code}",
                "details": response.text  # Fixed to use response.text instead of response.success
            }
            
    except Exception as e:
        # Handle exceptions (connection errors, etc.)
        return {
            "success": False,
            "message": "Request failed",
            "error": str(e)
        }

def fetch_skin_color(image_folder_path: str):
    # Prepare the request payload
    payload = {
        "input_dir": image_folder_path,
    }
    
    try:
        # Send the POST request to the endpoint
        response = requests.post(endpoint_url.get('color_endpoint_url'), json=payload)

        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            if result["success"] == True:  # Fixed dictionary access                            
                return result["rgbvalue"]
            return 136, 88, 67
        else:
            # Handle error responses
            return 136, 88, 67
            
    except Exception as e:
        # Handle exceptions (connection errors, etc.)
        return None, None, None

# Example usage
if __name__ == "__main__":
    image_folder_path = "./sample_data"
    # Example prompt
    prompt = "Victor sitting on the beach at sunset, looking out at the ocean, wearing a t-shirt and shorts, with a towel behind him."
    is_person = True
    output_dir = "./temp"
    skin_color = "224, 172, 105"
    image_style = "anime"
    id = 1
    # Optional additional parameters
        
    # Call the function and print the result
    value_r, value_g, value_b = fetch_skin_color(image_folder_path)
    skin_color = f"{value_r}, {value_g}, {value_b}"
    result = generate_image(
        id=id,
        prompt=prompt,
        is_person=is_person,
        skin_color=skin_color,  # Added missing parameter
        image_style='anime',
        output_dir=output_dir,
    )
        
    if result["success"]:
        print("✅ Image generation successful!")
    else:
        print("❌ Image generation failed.")
