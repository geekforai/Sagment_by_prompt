from PIL import Image, ImageDraw, ImageFont

def draw_bounding_shapes(image_path, shapes, labels):
    # Load the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Check that the number of shapes matches the number of labels
    if len(shapes) != len(labels):
        raise ValueError("The number of shapes must match the number of labels.")

    for shape, label in zip(shapes, labels):
        if len(shape) == 4:  # Assuming it's a bounding box
            x1, y1, x2, y2 = map(int, shape)
            # Draw the rectangle (bounding box)
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)  # Blue color, thickness 2

        else:  # Assuming it's a polygon
            # Draw the polygon
            draw.polygon(shape, outline="blue", fill=None)  # Blue outline

        # Add label if provided
        if label:
            # Use a specified font file and size
            font = ImageFont.truetype('arial.ttf', 30)
            # Get the bounding box for the text
            text_bbox = draw.textbbox((0, 0), label, font=font)  # The position can be any point
            
            # Calculate the position for the label
            text_x = shape[0][0] if len(shape) > 0 else 0  # For bounding box, just use x1
            text_y = shape[0][1] - text_bbox[3] if len(shape) > 0 else 0  # Offset for label position
            
            # Draw a background rectangle for the text
            draw.rectangle([text_x, text_y, text_x + text_bbox[2], text_y + text_bbox[3]], fill="blue")
            
            # Draw the label
            draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)

    # Show the image with the bounding shapes
    image.show()
import requests

def send_image_to_flask(image_path, text_input):
    url = 'https://1ec9-3-108-184-44.ngrok-free.app/process_image'  # URL of the Flask endpoint

    # Prepare the files and data
    files = {'image': open(image_path, 'rb')}  # Open the image file in binary mode
    data = {'text_input': text_input}

    # Send a POST request to the endpoint
    response = requests.post(url, files=files, data=data)
    print(response.json())
    # Check the response
    return response.json()





import cv2
import matplotlib.pyplot as plt
import numpy as np

# Sample JSON output


# Function to draw bounding boxes and polygons
def draw_shapes(image, json_output):
    for result in json_output['results']:
        bboxes = result['result']['bboxes']
        bboxes_labels = result['result']['bboxes_labels']
        polygons = result['result']['polygons']
        polygons_labels = result['result']['polygons_labels']

        # Draw bounding boxes
        for bbox, label in zip(bboxes, bboxes_labels):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw polygons
        for polygon, label in zip(polygons, polygons_labels):
            pts = np.array(polygon[0], np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
            # Optionally add label for polygon
            cv2.putText(image, label, (int(polygon[0][0]), int(polygon[0][1] - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image
if __name__ == '__main__':
    image_path = 'head-phones-template-design-ff6e8fcd8c23b654930b7e0628178971_screen.jpg'  # Replace with your image file path
    text_input = 'title,description-subheading,logo,product-earphone-headphone'  # Replace with your desired text input
    response =send_image_to_flask(image_path, text_input)

# Load your image
    image_path = 'head-phones-template-design-ff6e8fcd8c23b654930b7e0628178971_screen.jpg'
    image = cv2.imread(image_path)
    # Draw shapes on the image
    image_with_shapes = draw_shapes(image, response)

    # Convert BGR to RGB for displaying with matplotlib
    image_with_shapes_rgb = cv2.cvtColor(image_with_shapes, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.imshow(image_with_shapes_rgb)
    plt.axis('off')  # Hide axes
    plt.show()
