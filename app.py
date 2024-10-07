from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import subprocess
import os
import io
import gc
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file
from PIL import Image
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from utils.florence import load_florence_model, run_florence_inference, FLORENCE_OPEN_VOCABULARY_DETECTION_TASK
from OllamaServer import OllamaServer
from typing import Optional  # Import Optional here

server = OllamaServer(port=11434)
app = Flask(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Florence model
FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=DEVICE)

# Load ControlNet model and pipeline
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "ashllay/stable-diffusion-v1-5-archive", 
    controlnet=controlnet, 
    safety_checker=None, 
    torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# Load the inpainting pipeline
inpainting_pipeline = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
)
inpainting_pipeline.enable_model_cpu_offload()

def clear_cuda_cache():
    gc.collect()
    torch.cuda.empty_cache()

def resize_image(image, max_size_kb=1000):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_size_kb = img_bytes.tell() / 1024  # Size in KB

    while img_size_kb > max_size_kb:
        new_width = int(image.width * 0.9)
        new_height = int(image.height * 0.9)
        image = image.resize((new_width, new_height), Image.LANCZOS)

        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_size_kb = img_bytes.tell() / 1024

    return image

def create_mask(image, bboxes):
    mask = np.zeros((image.height, image.width), dtype=np.uint8)
    for bbox in bboxes:
        print(bbox)
        x1, y1, x2, y2 = map(int, bbox)  # Ensure integer values
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=cv2.FILLED)
    return Image.fromarray(mask)

@app.route('/process_image', methods=['POST'])
def process_image_endpoint():
    clear_cuda_cache()
    
    if 'image' not in request.files or 'text_input' not in request.form:
        return jsonify({'error': 'Image and text_input are required'}), 400

    image_file = request.files['image']
    text_inputs = [text.strip() for text in request.form['text_input'].split(',')]
    image_path = f'temp_{image_file.filename}'
    image_file.save(image_path)

    results = []
    for text in text_inputs:
        result = process_image(image_path, text)
        results.append({'text_input': text, 'result': result})

    os.remove(image_path)
    return jsonify({'message': 'Image processing completed', 'results': results}), 200

def process_image(image_path, text_input) -> Optional[dict]:
    clear_cuda_cache()
    print('Processing text input:', text_input)
    image = Image.open(image_path).convert("RGB")
    _, result = run_florence_inference(
        model=FLORENCE_MODEL,
        processor=FLORENCE_PROCESSOR,
        device=DEVICE,
        image=image,
        task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
        text=text_input
    )
    print('Detected bounding boxes:', result['<OPEN_VOCABULARY_DETECTION>'])
    return result['<OPEN_VOCABULARY_DETECTION>']

@app.route('/generate', methods=['POST'])
def generate_image():
    clear_cuda_cache()

    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    file = request.files['image']
    prompt = request.form.get('prompt', 'a bird')
    prompt = 'generate a high resolution image ' + prompt

    image = Image.open(file).convert("RGB")
    image = resize_image(image, 50)
    image = np.array(image)
    edges = cv2.Canny(image, 100, 200)
    edges_image = Image.fromarray(np.concatenate([edges[:, :, None]] * 3, axis=2))

    generated_image = pipe(
        prompt,
        edges_image,
        negative_prompt="blurry, disfigured, low quality",
        num_inference_steps=30,
        guidance_scale=12,
        height=768,
        width=768,
        generator=torch.manual_seed(42)
    ).images[0]

    img_io = io.BytesIO()
    generated_image.save(img_io, format='PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png', as_attachment=True, download_name='generated_image.png')

@app.route('/inpaint', methods=['POST'])
def inpaint():
    clear_cuda_cache()

    if 'image' not in request.files or 'bboxes' not in request.form:
        return jsonify({'error': 'Image and bounding boxes are required.'}), 400
    
    file = request.files['image']
    init_image = Image.open(file.stream).convert("RGB")

    try:
        bboxes = eval(request.form['bboxes'])  # Expecting a list of tuples
    except Exception:
        return jsonify({'error': 'Invalid bounding boxes format.'}), 400

    mask_image = create_mask(init_image, bboxes)

    prompt = "a black cat with glowing eyes, cute, adorable, disney, pixar, highly detailed, 8k"
    negative_prompt = "bad anatomy, deformed, ugly, disfigured"

    image = inpainting_pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask_image).images[0]

    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, port=8000)
