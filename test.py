from PIL import Image, ImageDraw

def draw_bounding_box(image_path, box):
    # Load the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Extract coordinates
    x1, y1, x2, y2 = map(int, box)
    
    # Draw the rectangle (bounding box)
    draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)  # Blue color, thickness 2

    # Show the image with the bounding box
    image.show()

import requests

def send_image_to_flask(image_path, text_input):
    url = 'http://127.0.0.1:5000/process_image'  # URL of the Flask endpoint

    # Prepare the files and data
    files = {'image': open(image_path, 'rb')}  # Open the image file in binary mode
    data = {'text_input': text_input}

    # Send a POST request to the endpoint
    response = requests.post(url, files=files, data=data)

    # Check the response
    if response.status_code == 202:
        print("Image processing started. Check the server for results.")
    else:
        print(f"Error: {response.status_code}, {response.json()}")

# Example usage
# if __name__ == '__main__':
#     image_path = '20jan9.jpg'  # Replace with your image file path
#     text_input = 'description,title,product'  # Replace with your desired text input
#     send_image_to_flask(image_path, text_input)
