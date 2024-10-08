from flask import Flask, request, jsonify, send_file
from diffusers import (AutoPipelineForInpainting, 
                       StableDiffusionControlNetPipeline, 
                       ControlNetModel,AutoencoderKL,StableDiffusionXLControlNetPipeline, 
                       UniPCMultistepScheduler)
from utils.florence import (load_florence_model, 
                             run_florence_inference, 
                             FLORENCE_OPEN_VOCABULARY_DETECTION_TASK)
import ollama
import base64
import io
import os
import gc
import numpy as np
import cv2
import subprocess
from PIL import Image
import torch
from typing import Optional
import image_utils
from model import Model
# Initialize Flask app
app = Flask(__name__)
model = Model(base_model_id="ashllay/stable-diffusion-v1-5-archive")
# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=DEVICE)
# Load models
# Utility functions
#controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
#pipe = StableDiffusionControlNetPipeline.from_pretrained(
 #   "ashllay/stable-diffusion-v1-5-archive", 
  #  controlnet=controlnet, 
   # safety_checker=None, 
   # torch_dtype=torch.float16
#)
#pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
#pipe.enable_model_cpu_offload()
def clear_cuda_cache():
    """Clear CUDA cache to free up memory."""
    gc.collect()
    torch.cuda.empty_cache()
import subprocess

def force_kill_process_by_name(process_name):
    """Forcefully kill all processes with the given name."""
    try:
        # Find the PIDs of the processes
        result = subprocess.run(
            ['pgrep', process_name],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"No matching processes found for '{process_name}'.")
            return

        # Get the list of PIDs
        pids = result.stdout.strip().split('\n')

        if not pids:
            print(f"No matching processes found for '{process_name}'.")
            return

        # Kill each process
        for pid in pids:
            subprocess.run(['sudo', 'kill', '-9', pid])

        print(f"Forcefully killed all instances of '{process_name}'.")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def resize_image(image, max_size_kb=300):
    clear_cuda_cache()
    """Resize image to ensure it is under the specified size."""
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    
    while img_bytes.tell() / 1024 > max_size_kb:
        new_size = (int(image.width * 0.9), int(image.height * 0.9))
        image = image.resize(new_size, Image.LANCZOS)
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')

    return image

# API endpoints
@app.route('/generate-prompt', methods=['POST'])
def generate_prompt():
    clear_cuda_cache()
    data = request.json
    prompt = data.get('prompt', '')
   
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    
    prompt = f"write a prompt to generate  {prompt}.  it should contain 100 word only and write only the prompt"

    try:
        response = ollama.chat(model='llama3.1', messages=[{'role': 'user', 'content': prompt}])
        title = response['message']['content']
        force_kill_process_by_name('ollama_llama_se')
        force_kill_process_by_name('ollama_llama_se')
        print(title)
        return jsonify({'title': title})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate-title', methods=['POST'])
def generate_title():
    clear_cuda_cache()
    data = request.json
    prompt = data.get('prompt', '')

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    
    prompt = f"""
    You are generating a social media post for {prompt}. 
    **Title**: Provide a bold, uppercase, attention-grabbing title (under 50 characters).
    **Subheading**: Provide a brief, engaging subheading (1 sentence) that complements the title and drives interaction.
    **Hashtags**: Generate 1-3 relevant hashtags to enhance discoverability.
    Example Format: **Title**: HUGE DISCOUNT ON SMARTWATCHES - ONLY $99!
    **Subheading**: Get the latest smartwatch at an unbeatable price. Limited stock available!
    **Hashtags**: #SmartwatchDeal, #TechSale, #WearableTech, #SmartwatchLovers
    """

    try:
        response = ollama.chat(model='llama3.1', messages=[{'role': 'user', 'content': prompt}])
        title = response['message']['content']
        #force_kill_process_by_name('ollama_llama_se')
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
    #force_kill_process_by_name('ollama_llama_server')
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
    #force_kill_process_by_name('ollama_llama_se')
    print('Detected bounding boxes:', result['<OPEN_VOCABULARY_DETECTION>'])
    return result['<OPEN_VOCABULARY_DETECTION>']
import random
@app.route('/generate', methods=['POST'])
def generate_image():
    clear_cuda_cache()

    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    file = request.files['image']
    prompt = request.form.get('prompt', 'a bird')
    prompt = 'Generate a high-resolution image ' + prompt + ' and background not be blank, fill it with related textures ' 

    image = Image.open(file).convert("RGB")
    image = resize_image(image, 50)
    image = np.array(image)
    edges = cv2.Canny(image, 100, 200)
    edges_image = Image.fromarray(np.concatenate([edges[:, :, None]] * 3, axis=2))
    seed=random.choice(range(0,2147483647))
    print(seed)



    #prompt = "A beautiful landscape"
    additional_prompt = "best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    num_images = 1
    image_resolution = 768  # Ensure this is within MAX_IMAGE_RESOLUTION
    num_steps = 20
    guidance_scale = 9
    low_threshold = 100  # Canny edge detection parameters
    high_threshold = 200


    #generated_image = pipe(
     #   prompt,
     #   edges_image,
     #   negative_prompt="blurry, disfigured,longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality,  low quality,bad human body parts",
     #   num_inference_steps=30,
     #   additional_prompt='best quality, extremely detailed',
     #   guidance_scale=9,
     #   height=768,
     #   width=768,
     #   generator=torch.manual_seed(2147483647)
    #).images[0]
    generated_image = model.process_canny(
        image=image_array,
        prompt=prompt,
        additional_prompt=additional_prompt,
        negative_prompt=negative_prompt,
        num_images=num_images,
        image_resolution=image_resolution,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )

    #force_kill_process_by_name('ollama_llama_se')

    generated_image_path = 'generated_image.png'
    generated_image.save(generated_image_path)
    clear_cuda_cache()
    return send_file(generated_image_path, mimetype='image/png', as_attachment=True, download_name='generated_image.png')
@app.route('/paste_image', methods=['POST'])
def paste_image():
    clear_cuda_cache()
    # Retrieve the images from the request files
    base_image_file = request.files['base_image']
    paste_image_file = request.files['paste_image']
    
    # Open the images using PIL
    base_image = Image.open(base_image_file)
    paste_image = Image.open(paste_image_file)
    
    # Get the bounding box from the form data
    bbox = [ int(float(x)) for x in request.form.getlist('bbox')]
    print(bbox)
    #bbox = tuple(map(int, bbox.split(',')))  # Convert bbox string to tuple of integers

    # Call the utility function to paste the image
    result_image = image_utils.paste_image_on_background(base_image, paste_image, bbox)
    
    # Save the result image to a BytesIO object
    img_io = io.BytesIO()
    result_image.save(img_io, 'PNG')
    img_io.seek(0)
    #force_kill_process_by_name('ollama_llama_se')
    # Send the resulting image back
    return send_file(img_io, mimetype='image/png')
@app.route('/inpaint', methods=['POST'])
def inpaint():
    clear_cuda_cache()
    # Retrieve the image from the request files
    image_file = request.files['image']
    
    # Open the image using PIL
    pil_image = Image.open(image_file)
    
    # Get bounding boxes from the form data
    bounding_boxes =[int(float(num)) for num in request.form.getlist('bounding_boxes')]
    print(bounding_boxes)
    #bounding_boxes = eval(bounding_boxes)  # Convert string representation of list to actual list
    inpaint_radius = int(request.form.get('inpaint_radius', 3))  # Default radius is 3
    print(bounding_boxes)
    # Call the utility function to inpaint the image
    result_image = image_utils.inpaint_with_bboxes(pil_image, bounding_boxes, inpaint_radius)
    
    # Save the result image to a BytesIO object 
    #force_kill_process_by_name('ollama_llama_se')
    img_io = io.BytesIO()
    result_image.save(img_io, 'PNG')
    img_io.seek(0)

    # Send the resulting image back
    return send_file(img_io, mimetype='image/png')
@app.route('/remove_text', methods=['POST'])
def remove_text():
    clear_cuda_cache()
    # Retrieve the image from the request files
    image_file = request.files['image']
    
    # Open the image using PIL
    image = Image.open(image_file)

    # Call the utility function to remove text from the image
    result_image = image_utils.remove_text_from_image(image)
    
    # Save the result image to a BytesIO object
    img_io = io.BytesIO()
    result_image.save(img_io, 'PNG')
    img_io.seek(0)
    
    # Send the resulting image back
    #force_kill_process_by_name('ollama_llama_se')
    return send_file(img_io, mimetype='image/png' )

@app.route('/draw_text', methods=['POST'])
def draw_text():
    clear_cuda_cache()
    # Retrieve the image from the request files
    image_file = request.files['image']
    
    # Open the image using PIL
    image = Image.open(image_file)

    # Get the text and other parameters from the form data
    text = request.form['text']
    bbox = eval(request.form['bbox'])  # Convert string representation to tuple/list
    font_path = request.form.get('font_path', "arial.ttf")
    gradient_start = tuple(request.form.get('gradient_start', (100, 0, 0)))
    gradient_end = tuple(request.form.get('gradient_end', (0, 0, 100)))

    # Call the utility function to draw text in the image
    result_image = image_utils.draw_multiline_text_in_bbox(image, text, bbox, font_path, gradient_start, gradient_end)
    
    # Save the result image to a BytesIO object
    img_io = io.BytesIO()
    result_image.save(img_io, 'PNG')
    img_io.seek(0)
    
    # Send the resulting image back
    return send_file(img_io, mimetype='image/png')
# Main entry point
if __name__ == '__main__':
    app.run(debug=True, port=8000)
