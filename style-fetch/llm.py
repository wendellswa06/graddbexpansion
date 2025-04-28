import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

IMAGE_STYLES = [
    "Watercolor Painting",
    "Oil Painting",
    "Digital Art",
    "Pencil Sketch",
    "Comic Book Style",
    "Cyberpunk",
    "Steampunk",
    "Impressionist",
    "Pop Art",
    "Minimalist",
    "Gothic",
    "Art Nouveau",
    "Pixel Art",
    "Anime",
    "3D Render",
    "Low Poly",
    "Photorealistic",
    "Vector Art",
    "Abstract Expressionism",
    "Realism",
    "Futurism",
    "Cubism",
    "Surrealism",
    "Baroque",
    "Renaissance",
    "Fantasy Illustration",
    "Sci-Fi Illustration",
    "Ukiyo-e",
    "Line Art",
    "Black and White Ink Drawing",
    "Graffiti Art",
    "Stencil Art",
    "Flat Design",
    "Isometric Art",
    "Retro 80s Style",
    "Vaporwave",
    "Dreamlike",
    "High Fantasy",
    "Dark Fantasy",
    "Medieval Art",
    "Art Deco",
    "Hyperrealism",
    "Sculpture Art",
    "Caricature",
    "Chibi",
    "Noir Style",
    "Lowbrow Art",
    "Psychedelic Art",
    "Vintage Poster",
    "Manga",
    "Holographic",
    "Kawaii",
    "Monochrome",
    "Geometric Art",
    "Photocollage",
    "Mixed Media",
    "Ink Wash Painting",
    "Charcoal Drawing",
    "Concept Art",
    "Digital Matte Painting",
    "Pointillism",
    "Expressionism",
    "Sumi-e",
    "Retro Futurism",
    "Pixelated Glitch Art",
    "Neon Glow",
    "Street Art",
    "Acrylic Painting",
    "Bauhaus",
    "Flat Cartoon Style",
    "Carved Relief Art",
    "Fantasy Realism",
]

def detect_image_style_with_llm(prompt, image_styles=IMAGE_STYLES, max_styles=2):
    """
    Detect image styles from a prompt using an open-source LLM.
    
    Args:
        prompt (str): The input prompt for image generation.
        image_styles (list): List of possible image styles.
        max_styles (int): Maximum number of styles to detect.
        
    Returns:
        list: Detected image styles (up to max_styles).
    """
    # Load model and tokenizer
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # You can use a smaller model if needed
    
    # Use device with CUDA if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to smaller model...")
        # Fallback to a smaller model if the first one fails
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
    
    # Format the list of styles for better processing
    styles_text = "\n".join([f"- {style}" for style in image_styles])
    
    # Create the prompt for the LLM
    llm_prompt = f"""
Below is a list of image styles:

{styles_text}

Based on the following prompt: "{prompt}"

Identify up to {max_styles} image styles from the list above that best match the prompt.
Return only the exact names of the styles, separated by commas, without any additional text.
If no styles match, return "No specific style detected".

Image styles detected:
"""
    
    # Generate response from the model
    inputs = tokenizer(llm_prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        temperature=0.1,  # Low temperature for more deterministic outputs
        do_sample=False,  # Disable sampling for deterministic generation
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Get the generated text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the part after our input prompt
    response = response[len(llm_prompt):].strip()
    
    # Parse the response to get the styles
    if "No specific style detected" in response:
        return []
    
    # Extract style names from the response
    detected_styles = []
    for style in image_styles:
        if style.lower() in response.lower():
            detected_styles.append(style)
            if len(detected_styles) >= max_styles:
                break
    
    # If we didn't find styles using exact matching, try comma splitting
    if not detected_styles:
        potential_styles = [s.strip() for s in response.split(',')]
        for potential in potential_styles:
            for style in image_styles:
                if potential.lower() == style.lower():
                    detected_styles.append(style)
                    if len(detected_styles) >= max_styles:
                        break
            if len(detected_styles) >= max_styles:
                break
    
    return detected_styles


# Function for offline usage without requiring a model
def detect_image_style_offline(prompt, image_styles=IMAGE_STYLES, max_styles=2):
    """
    Fallback function for detecting image styles without using an LLM.
    Uses keyword matching as a simpler approach.
    """
    prompt_lower = prompt.lower()
    detected_styles = []
    
    # First pass: Check for exact style matches
    for style in image_styles:
        style_lower = style.lower()
        if style_lower in prompt_lower:
            detected_styles.append(style)
            if len(detected_styles) >= max_styles:
                return detected_styles
    
    # Second pass: Check for partial matches if needed
    if len(detected_styles) < max_styles:
        for style in image_styles:
            if style not in detected_styles:  # Skip already detected styles
                style_lower = style.lower()
                words = style_lower.split()
                
                # Filter for significant words
                significant_words = [word for word in words if len(word) > 3]
                if not significant_words:
                    significant_words = words
                
                # Check if any significant words match
                if any(word in prompt_lower for word in significant_words):
                    detected_styles.append(style)
                    if len(detected_styles) >= max_styles:
                        break
    
    return detected_styles


# Example usage
if __name__ == "__main__":
    test_prompts = [
        "A realistic depiction of a steampunk airship with intricate gears and brass fittings, floating over a foggy Victorian cityscape with detailed textures and natural lighting.",
        "A steampunk-inspired mechanical automaton, realistically rendered with exposed clockwork mechanisms, polished brass, and aged leather components, set against a dimly lit workshop with scattered tools.",
        "A detailed steampunk train station with steam billowing from a coal-fired locomotive, vintage advertisements on the walls, and passengers dressed in period attire, all rendered in high realism with accurate shadows and textures.",
        "A steampunk laboratory filled with bubbling flasks, brass microscopes, and steam-powered machinery, realistically portrayed with attention to the sheen of polished metal and the glow of warm incandescent lighting.",
        "A steampunk fashion model wearing a corset adorned with gears and chains, leather gloves, and goggles, standing in a rustic forest setting with dappled sunlight filtering through the trees, capturing both steampunk and realistic elements.",
        "A steampunk airship hangar with massive, steam-driven cranes and workers in overalls, realistically depicted with the metallic textures, steam clouds, and the industrial ambiance of the setting.",
        "A steampunk submarine rising from the ocean depths, its hull covered in barnacles and seaweed, with brass portholes and steam escaping from its engines, rendered in high realism with underwater lighting effects.",
        "A steampunk workshop with a large steam-powered printing press, piles of paper, and an array of tools, realistically portrayed with the metallic sheen, the glow of a kerosene lamp, and the intricate details of machinery.",
        "A steampunk clock tower with gears visible through its windows, ornate brass decorations, and a clock face showing the time, set against a sunset with warm, realistic lighting and shadow effects.",
        "A steampunk automaton playing a piano, its gears and springs visible, with the keys illuminated by soft, diffused light, realistically rendered with attention to the mechanical details and the musical setting.",
        "A steampunk dirigible with a glass observation deck, passengers in period attire, and the city skyline below, depicted in high realism with the shimmer of polished metal and the clarity of distant buildings.",
    ]
    
    try:
        # Try to use the LLM approach first
        new_prompt = ""
        for prompt in test_prompts:
            new_prompt += prompt
        prompt = new_prompt
        styles = detect_image_style_with_llm(prompt)
        print(f"Prompt: {prompt}")
        print(f"Detected styles (LLM): {styles}\n")
    except Exception as e:
        print(f"Error using LLM: {e}")
        print("Falling back to offline detection method...")
        
        # Fall back to the offline approach if the LLM fails
        for prompt in test_prompts:
            styles = detect_image_style_offline(prompt)
            print(f"Prompt: {prompt}")
            print(f"Detected styles (Offline): {styles}\n")
