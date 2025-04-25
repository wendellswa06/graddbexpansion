import argparse
import os
import time
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from diffusers import FluxPipeline  # Most likely import path

class RequestData(BaseModel):
    id: int
    is_person: bool
    prompt: str = Field(default="A man wearing a white shirt")
    output_dir: str = Field(default="./")
    skin_color: str = Field(default="white")
    image_style: str = Field(default="anime")

app = FastAPI()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--port", default=8093, type=int)
    return parser.parse_args()

args = get_args()

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power


@app.post("/generate")
async def text_to_3d(data: RequestData):
    for item in os.listdir(data.output_dir):
        item_path = os.path.join(data.output_dir, item)
        if os.path.isfile(item_path) and item == f'{data.id}.png':
            os.unlink(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

    is_person = data.is_person
    if is_person:
        skin_color = data.skin_color
        extra_prompt = f"Face skin color's RGB values are {skin_color}"
    else:
        image_style = data.image_style
        extra_prompt = f'{image_style} style'
    
    prompt = data.prompt
    prompt = f"{prompt}, {extra_prompt}"
    output_dir = data.output_dir    
    os.makedirs(output_dir, exist_ok=True)

    start = time.time()
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=0.0,
        num_inference_steps=2,
        max_sequence_length=256,
        generator=torch.Generator("cuda").manual_seed(0)
    ).images[0]

    # Save image to the specified output directory
    output_path = os.path.join(output_dir, f"{data.id}.png")
    image.save(output_path)
    
    print(f"Successfully generated: {output_dir}")
    print(f"Generation time: {time.time() - start}")
    return {"success": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
