from typing import Optional
import threading
from flask import Flask, request, jsonify
from PIL import Image
import supervision as sv
import torch
from PIL import Image
import os
from utils.florence import load_florence_model, run_florence_inference, \
    FLORENCE_OPEN_VOCABULARY_DETECTION_TASK
from utils.sam import load_sam_image_model, run_sam_inference

DEVICE = torch.device("cuda")
# DEVICE = torch.device("cpu")

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=DEVICE)
SAM_IMAGE_MODEL = load_sam_image_model(device=DEVICE)


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def process_image(image_path, text_input) -> Optional[Image.Image]:
    print('start ',text_input)
    image=Image.open(image_path)
    _, result = run_florence_inference(
        model=FLORENCE_MODEL,
        processor=FLORENCE_PROCESSOR,
        device=DEVICE,
        image=image,
        task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
        text=text_input
    )
    print('result of bbox ',result['<OPEN_VOCABULARY_DETECTION>'])
    return result['<OPEN_VOCABULARY_DETECTION>']
app = Flask(__name__)
@app.route('/process_image', methods=['POST'])
def process_image_endpoint():
    # Check if the image and text_input are provided
    if 'image' not in request.files or 'text_input' not in request.form:
        return jsonify({'error': 'Image and text_input are required'}), 400

    # Get the uploaded image
    image_file = request.files['image']
    text_input = request.form['text_input']

    # Split the text input by commas to handle multiple inputs
    text_inputs = [text.strip() for text in text_input.split(',')]

    # Save the image temporarily
    image_path = f'temp_{image_file.filename}'
    image_file.save(image_path)

    # Create a list to hold results
    results = []

    # Define a function to process the image for each text input
    def process_and_return_result(image_path, text_input):
        # Process the image
        result = process_image(image_path, text_input)
        results.append({'text_input': text_input, 'result': result})

    # Start a thread for each text input
    threads = []
    for text in text_inputs:
        thread = threading.Thread(target=process_and_return_result, args=(image_path, text))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # Clean up: remove the temporary image file
    os.remove(image_path)

    return jsonify({'message': 'Image processing completed', 'results': results}), 200

if __name__ == '__main__':
    app.run(debug=True)