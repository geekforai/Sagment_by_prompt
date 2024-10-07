from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import psutil
import subprocess
from typing import Optional
import ollama
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
server = OllamaServer(port=11434)
# Initialize Flask app
app = Flask(__name__)

# Set device for PyTorch
DEVICE = torch.device("cuda")

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
# Set PyTorch optimizations
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def kill_process_by_name(process_name):
    try:
        # Construct the command to forcefully kill the process
        command = ['sudo', 'pkill', '-9', process_name]
        
        # Execute the command
        subprocess.run(command, check=True)
        print(f"Successfully forcefully killed the process: {process_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
def resize_image(image, max_size_kb=1000):
    """
    Resize a PIL Image until its size is less than or equal to max_size_kb,
    while maintaining the aspect ratio.

    Parameters:
    - image: PIL Image object
    - max_size_kb: int, maximum size in kilobytes (default is 700 KB)

    Returns:
    - PIL Image object, resized to meet the size requirement
    """
    # Convert the image to bytes to check its size
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_size_kb = img_bytes.tell() / 1024  # Size in KB

    # Loop to resize until the image size is within the limit
    while img_size_kb > max_size_kb:
        # Scale down the image by 10%
        new_width = int(image.width * 0.9)
        new_height = int(image.height * 0.9)

        # Resize the image
        image = image.resize((new_width, new_height), Image.LANCZOS)

        # Check the new size
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_size_kb = img_bytes.tell() / 1024

    return image
def process_image(image_path, text_input) -> Optional[dict]:
    gc.collect()

# Clear CUDA cache
    torch.cuda.empty_cache()
    """Process an image with the Florence model and return detected results."""
    print('Processing text input:', text_input)
    image = Image.open(image_path)
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

@app.route('/process_image', methods=['POST'])
def process_image_endpoint():
    gc.collect()

# Clear CUDA cache
    torch.cuda.empty_cache()
    """API endpoint to process an image with multiple text inputs."""
    if 'image' not in request.files or 'text_input' not in request.form:
        return jsonify({'error': 'Image and text_input are required'}), 400

    image_file = request.files['image']
    text_inputs = [text.strip() for text in request.form['text_input'].split(',')]
    image_path = f'temp_{image_file.filename}'
    image_file.save(image_path)

    results = []

    # Process each text input sequentially
    for text in text_inputs:
        result = process_image(image_path, text)
        results.append({'text_input': text, 'result': result})

    # Clean up temporary image file
    os.remove(image_path)

    return jsonify({'message': 'Image processing completed', 'results': results}), 200

@app.route('/generate', methods=['POST'])
def generate_image():

    gc.collect()

# Clear CUDA cache
    torch.cuda.empty_cache()
    cfg_scale = 12  # Adjust for better adherence to the prompt 
    height = 768  # High resolution
    width = 768   # High resolution
    seed = 42  # Set seed for reproducibility

# Generate the image
    generator = torch.manual_seed(seed) 
    negative_prompts = (
    " blurry, deformed, disfigured, duplicate, blurred "
    "extra arms, extra fingers, extra legs, fused fingers, gross proportions, jpeg artifacts, "
    "low quality, lowres, malformed limbs, missing arms, missing legs, morbid, mutation, "
    "out of frame, poorly drawn face, poorly drawn hands, signature, text, too many fingers, ugly, "
    "watermark, worst quality, blank background, boring background, body out of frame, "
    "disproportioned, distorted, duplicated features, flaw, grains, grainy, hazy, "
    "improper scale, incorrect ratio, kitsch, low contrast, macro, multiple views, "
    "overexposed, oversaturated, surreal, unfocused, unattractive, unnatural pose, "
    "3D, absent limbs, additional appendages, broken finger, broken hand, cartoon, "
    "deformed structures, low resolution, missing parts, off-center, out of focus, "
    "over-saturated color, poorly rendered, unrealistic, upside down"
    )    
#negative_prompts = "bad anatomy, bad proportions, blurry, cloned face, cropped, deformed, dehydrated, disfigured, duplicate, error, extra arms, extra fingers, extra legs, extra limbs, fused fingers, gross proportions, jpeg artifacts, long neck, low quality, lowres, malformed limbs, missing arms, missing legs, morbid, mutated hands, mutation, mutilated, out of frame, poorly drawn face, poorly drawn hands, signature, text, too many fingers, ugly, username, watermark, worst quality"
    """API endpoint to generate an image using Stable Diffusion."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    file = request.files['image']
    prompt = request.form.get('prompt', 'a bird')  # Default prompt
    prompt='generate a 8k quality (high resolution) image'+prompt

    # Process the image for Canny edge detection
    image = Image.open(file).convert("RGB")
    image = resize_image(image,50)
    image = np.array(image)
    edges = cv2.Canny(image, 100, 200)
    edges_image = Image.fromarray(np.concatenate([edges[:, :, None]] * 3, axis=2))

    # Generate image using the model
    #generated_image = pipe(prompt, edges_image,negative_prompt=negative_prompts, num_inference_steps=50).images[0]
    generated_image = pipe(
    prompt,
    edges_image,
    negative_prompt=negative_prompts,
    num_inference_steps=30,
    guidance_scale=cfg_scale,
    height=height,
    width=width,
    generator=generator
    ).images[0]
    # Save the generated image to a bytes buffer
    img_io = io.BytesIO()
    generated_image.save(img_io, format='PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png', as_attachment=True, download_name='generated_image.png')
@app.route('/generate_prompt', methods=['POST'])
def generate_prmpt():
    gc.collect()
    server.start()
# Clear CUDA cache
    torch.cuda.empty_cache()
    """API endpoint to generate text using the Ollama model."""
    data = request.get_json()

    # Check if 'prompt' is in the request data
    if 'prompt' not in data:
        return jsonify({'error': 'Prompt is required'}), 400
    prompt = data['prompt']
    template=f"""
    You are generating a pormpt to create the image using this {prompt}.
    generate only prompt beacuse it will directly pass to stable diffusion
    controlnet model and answer should contain  only 50 words and strictly 
    rely on edge image also do not create any other product which mentioned in prompt 
    """
    

  #  try:
        # Generate text using the Ollama model
    generated_text = ollama.generate(model='llama3.1', prompt=template)
    kill_process_by_name('ollama_llama_se')
    #kill_process_by_name('python')
#        server.kill_by_port()
    return jsonify({'generated_text': generated_text}), 200
@app.route('/generate_text', methods=['POST'])
def generate_text():
    gc.collect()
    server.start()
# Clear CUDA cache
    torch.cuda.empty_cache()
    """API endpoint to generate text using the Ollama model."""
    data = request.get_json()

    # Check if 'prompt' is in the request data
    if 'prompt' not in data:
        return jsonify({'error': 'Prompt is required'}), 400
    prompt = data['prompt']
    template=f"""
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
    

  #  try:
        # Generate text using the Ollama model
    generated_text = ollama.generate(model='llama3.1', prompt=template)
    kill_process_by_name('ollama_llama_se')
    #kill_process_by_name('python')
#        server.kill_by_port()
    return jsonify({'generated_text': generated_text}), 200
#    except Exception as e:

 #       return jsonify({'error': str(e)}), 500

pipeline = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
)
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

# Function to create a mask from bounding boxes
def create_mask(image, bboxes):
    mask = np.zeros((image.height, image.width), dtype=np.uint8)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=cv2.FILLED)
    return Image.fromarray(mask)

@app.route('/inpaint', methods=['POST'])
def inpaint():
    if 'image' not in request.files or 'bboxes' not in request.form:
        return jsonify({'error': 'Image and bounding boxes are required.'}), 400
    
    # Get the image
    file = request.files['image']
    init_image = Image.open(file.stream).convert("RGB")

    # Get bounding boxes from the request
    try:
        bboxes = eval(request.form['bboxes'])  # Expecting a list of tuples
    except Exception as e:
        return jsonify({'error': 'Invalid bounding boxes format.'}), 400
    
    # Create the mask
    mask_image = create_mask(init_image, bboxes)

    # Define the prompts
    prompt = "a black cat with glowing eyes, cute, adorable, disney, pixar, highly detailed, 8k"
    negative_prompt = "bad anatomy, deformed, ugly, disfigured"

    # Inpainting
    image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask_image).images[0]

    # Save the inpainted image to a BytesIO object
    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True,port=8000)
