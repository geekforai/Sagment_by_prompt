from diffusers import AutoPipelineForInpainting
import ollama
import base64 
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
import image_utils

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
def clear_cuda_cache():
    gc.collect()
    torch.cuda.empty_cache()

def force_kill_process_by_name(process_name):
    """
    Forcefully kills all processes with the given name on Ubuntu.

    Parameters:
    - process_name (str): The name of the process to kill (e.g., 'firefox').

    Returns:
    - str: A message indicating the result of the operation.
    """
    try:
        # Use pkill with the -9 option to forcefully kill the process
        result = subprocess.run(
            ['pkill', '-9', process_name],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            return f"Forcefully killed all instances of '{process_name}'."
        else:
            return f"Error: {result.stderr.strip() or 'No matching processes found.'}"
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

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
@app.route('/generate-prompt', methods=['POST'])
def generate_prompt():
    # Get the prompt from the request
    data = request.json
    prompt = data.get('prompt', '')
    prompt=f"""
     generate a prompt to create a image using this info {prompt} and generate only prompt becouse it will be directly pass to image generation llm and length should be 60 words only
"""
     # Check if prompt is provided
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    try:
        # Call the Ollama chat model
        response = ollama.chat(model='llama3.1', messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])
        
        title = response['message']['content']
        print(title)
        force_kill_process_by_name('ollama_llama_se')
        return jsonify({'title': title})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/generate-title', methods=['POST'])
def generate_title():
    # Get the prompt from the request
    data = request.json
    prompt = data.get('prompt', '')
    prompt=f"""
     You are generating a social media post for {prompt}. 
    
     Please follow the format below:
    
     **Title**: Provide a bold, uppercase, attention-grabbing title (under 50 characters).
    
     **Subheading**: Provide a brief, engaging subheading (1 sentences) that complements the title and drives interaction.
    
     **Hashtags**: Generate 1-3 relevant hashtags to enhance discoverability.
    
     Example Format:
    
     **Title**: HUGE DISCOUNT ON SMARTWATCHES - ONLY $99!
    
     **Subheading**: Get the latest smartwatch at an unbeatable price. Limited stock available!
    
     **Hashtags**: #SmartwatchDeal, #TechSale, #WearableTech, #SmartwatchLovers
    
     Please follow this format and respond accordingly.
     """
     # Check if prompt is provided
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    try:
        # Call the Ollama chat model
        response = ollama.chat(model='llama3.1', messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])
        
        title = response['message']['content']
        print(title)
        return jsonify({'title': title})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
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
        negative_prompt="blurry, disfigured, low quality,bad human body parts",
        num_inference_steps=30,
        guidance_scale=12,
        height=768,
        width=768,
        generator=torch.manual_seed(42)
    ).images[0]
    os.remove('geneated_image.png')
    #img_io = io.BytesIO()
    generated_image.save('geneated_image.png')
    #img_io.seek(0)
    clear_cuda_cache()
    return send_file('geneated_image.png', mimetype='image/png', as_attachment=True, download_name='generated_image.png')

@app.route('/paste_image', methods=['POST'])
def paste_image():
    data = request.json
    base_image_bytes = data['base_image']    
    base_image_bytes = base64.b64decode(base_image_bytes)

    # Convert the bytes to a PIL Image
    base_image = Image.open(io.BytesIO(base_image_bytes))
    paste_image_bytes = data['paste_image']    
    paste_image_bytes = base64.b64decode(paste_image_bytes)

    # Convert the bytes to a PIL Image
    paste_image = Image.open(io.BytesIO(paste_image_bytes))

    bbox = data['bbox']
    
    result_image = image_utils.paste_image_on_background(base_image, paste_image, bbox)
    
    img_io = io.BytesIO()
    result_image.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

@app.route('/inpaint', methods=['POST'])
def inpaint():
    data = request.json
    image_bytes = data['image']    
    image_bytes = base64.b64decode(image_bytes)

    # Convert the bytes to a PIL Image
    pil_image = Image.open(io.BytesIO(image_bytes))
    bounding_boxes = data['bounding_boxes']
    inpaint_radius = data.get('inpaint_radius', 3)
    
    result_image = image_utils.inpaint_with_bboxes(pil_image, bounding_boxes, inpaint_radius)
    
    img_io = io.BytesIO()
    result_image.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')
@app.route('/remove_text', methods=['POST'])
def remove_text():
    data = request.json
    image_bytes = data['image']  # Expecting image bytes in the request
    image_bytes = base64.b64decode(image_bytes)
    # Convert the image bytes to a PIL Image
    image = Image.open(io.BytesIO(image_bytes))

    # Process the image to remove text
    result_image = image_utils.remove_text_from_image(image)

    # Prepare the response
    img_io = io.BytesIO()
    result_image.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')
@app.route('/draw_text', methods=['POST'])
def draw_text():
    data = request.json
    image_bytes = data['image']    
    image_bytes = base64.b64decode(image_bytes)

    # Convert the bytes to a PIL Image
    image = Image.open(io.BytesIO(image_bytes))

    text = data['text']
    bbox = data['bbox']
    font_path = data.get('font_path', "arial.ttf")
    gradient_start = tuple(data.get('gradient_start', (100, 0, 0)))
    gradient_end = tuple(data.get('gradient_end', (0, 0, 100)))

    result_image = image_utils.draw_multiline_text_in_bbox(image, text, bbox, font_path, gradient_start, gradient_end)

    img_io = io.BytesIO()
    result_image.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True, port=8000)

