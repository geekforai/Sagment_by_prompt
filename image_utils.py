from PIL import Image
def paste_image_on_background(base_image:Image, paste_image:Image, bbox)->Image:
    """
    Pastes one PIL image onto another using the provided bounding box.
    Preserves the aspect ratio of the pasted image.

    Args:
        base_image (Image): The base image as a PIL Image object.
        paste_image (Image): The image to paste as a PIL Image object with transparency.
        bbox (list): Bounding box [x1, y1, x2, y2] where:
            - (x1, y1) is the top-left corner.
            - (x2, y2) is the bottom-right corner.

    Returns:
        Image: The combined image with the pasted image.
    """
    # Validate the bounding box
    if len(bbox) != 4 or any(not isinstance(i, (int, float)) for i in bbox):
        raise ValueError("Bounding box must be a list of four numeric values [x1, y1, x2, y2].")

    # Calculate the position to paste the image (top-left corner of the bounding box)
    x1, y1, x2, y2 = map(int, bbox)  # Convert to integers
    paste_position = (x1, y1)

    # Get the size of the bounding box
    bbox_width = x2 - x1
    bbox_height = y2 - y1

    # Get the original size of the paste image
    paste_width, paste_height = paste_image.size

    # Calculate the aspect ratios
    aspect_ratio = paste_width / paste_height
    bbox_aspect_ratio = bbox_width / bbox_height

    # Determine new size while maintaining aspect ratio
    if bbox_aspect_ratio > aspect_ratio:
        # Bounding box is wider than the pasted image
        new_height = bbox_height
        new_width = int(new_height * aspect_ratio)
    else:
        # Bounding box is taller than the pasted image
        new_width = bbox_width
        new_height = int(new_width / aspect_ratio)

    # Resize the pasted image to fit the bounding box while maintaining aspect ratio
    paste_image = paste_image.resize((new_width, new_height), Image.LANCZOS)

    # Ensure the pasted image is in 'RGBA' mode to preserve transparency
    if paste_image.mode != 'RGBA':
        paste_image = paste_image.convert('RGBA')

    # Check if the pasted image fits within the base image dimensions
    if (x1 < 0 or y1 < 0 or x2 > base_image.width or y2 > base_image.height):
        raise ValueError("Bounding box is out of base image bounds.")

    # Paste the image onto the base image using the alpha channel as a mask
    base_image.paste(paste_image, paste_position, mask=paste_image)

    return base_image

import cv2
import numpy as np

def inpaint_with_bboxes(pil_image, bounding_boxes, inpaint_radius=1):
    """
    Inpaints the areas of a PIL image specified by bounding boxes.

    Parameters:
    - pil_image: PIL.Image, the input image
    - bounding_boxes: list of tuples, each tuple is (x, y, width, height)
    - inpaint_radius: int, radius for inpainting (default is 3)

    Returns:
    - inpainted_image: PIL.Image, the inpainted image
    """
    # Convert PIL image to NumPy array (OpenCV format)
    image = np.array(pil_image)

    # Check if the image is in RGB format
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError("Input image should be in RGB format.")

    # Create a mask with the same dimensions as the image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Fill the mask for each bounding box
    x,y,w,h=bounding_boxes
    cv2.rectangle(mask, (int(x), int(y)), (int(x + w), int(y + h)), 255, -1)  # Fill with white (255)

    # Inpaint the image using the mask
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)

    # Convert back to PIL Image
    inpainted_image_pil = Image.fromarray(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))

    return inpainted_image_pil

#**Remove the text from images**
import cv2
import numpy as np
import easyocr
from PIL import Image

def remove_text_from_image(pil_image: Image) -> Image:
    """
    Removes text from a given PIL image using EasyOCR and OpenCV inpainting.

    Args:
        pil_image (Image): The input PIL Image from which to remove text.

    Returns:
        Image: A PIL Image with the text removed.
    """
    # Convert PIL Image to NumPy array (OpenCV format)
    image = np.array(pil_image)

    # Initialize the EasyOCR reader
    reader = easyocr.Reader(['en'])
    
    # Detect text
    results = reader.readtext(image)

    # Create a mask for the detected text
    mask = np.zeros(image.shape, dtype=np.uint8)

    for (bbox, text, prob) in results:
        # Extract the bounding box coordinates
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        # Draw a filled rectangle on the mask
        cv2.rectangle(mask, top_left, bottom_right, (255, 255, 255), thickness=cv2.FILLED)

    # Invert the mask
    mask_inv = cv2.bitwise_not(mask)

    # Create a background using the mask
    background = cv2.bitwise_and(image, mask_inv)

    # Use inpainting to remove text
    inpainted_image = cv2.inpaint(background, cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # Convert the result back to a PIL Image
    inpainted_image_rgb = cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB)
    pil_image_result = Image.fromarray(inpainted_image_rgb)

    return pil_image_result


from PIL import Image, ImageDraw, ImageFont
def draw_multiline_text_in_bbox(image: Image.Image, text: str, bbox: tuple, 
                                  font_path: str = "arial.ttf", 
                                  gradient_start: tuple = (100, 0, 0), 
                                  gradient_end: tuple = (0, 0, 100)) -> Image.Image:
    """
    Draws multiline text inside a given bounding box on the provided image with a 3D effect and gradient color.
    Args:
        image (Image.Image): The image to draw on.
        text (str): The text to be drawn, which can contain line breaks.
        bbox (tuple): The bounding box as (left, upper, right, lower).
        font_path (str): The path to the TTF font file to be used.
        gradient_start (tuple): The RGB color to start the gradient (default red).
        gradient_end (tuple): The RGB color to end the gradient (default blue).
    Returns:
        Image.Image: The image with the text drawn inside the bounding box.
    """
    # Create a drawing context
    draw = ImageDraw.Draw(image)
    # Extract bounding box coordinates
    left, upper, right, lower = bbox
    bbox_width = right - left
    bbox_height = lower - upper
    # Set initial font size
    font_size = bbox_height // 2  # Start with a reasonable size

    # Load font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    # Reduce font size until the text fits in the bounding box
    while True:
        # Split the text into lines that fit within the bounding box width
        lines = []
        words = text.split()
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip()
            text_bbox = draw.textbbox((left, upper), test_line, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            if text_width <= bbox_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)  # Add the last line

        # Calculate total text height
        total_text_height = sum(draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in lines)
        if total_text_height <= bbox_height:
            break
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)

    # Calculate starting y position to center the text vertically in the bbox
    y = upper + (bbox_height - total_text_height) / 2

    # Draw each line of text with 3D effect and gradient color
    for i, line in enumerate(lines):
        # 3D effect: draw shadow first
        shadow_offset = 2  # Change this for more or less shadow
        shadow_color = (50, 50, 50)  # Dark gray shadow color
        shadow_text_bbox = draw.textbbox((left + shadow_offset, y + shadow_offset), line, font=font)
        shadow_width = shadow_text_bbox[2] - shadow_text_bbox[0]
        shadow_x = left + (bbox_width - shadow_width) / 2 + shadow_offset
        draw.text((shadow_x, y + shadow_offset), line, font=font, fill=shadow_color)

        # Calculate gradient color for the line
        r = int(gradient_start[0] + (gradient_end[0] - gradient_start[0]) * (i / len(lines)))
        g = int(gradient_start[1] + (gradient_end[1] - gradient_start[1]) * (i / len(lines)))
        b = int(gradient_start[2] + (gradient_end[2] - gradient_start[2]) * (i / len(lines)))
        gradient_color = (r, g, b)

        # Draw the main text with gradient color
        text_bbox = draw.textbbox((left, y), line, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        x = left + (bbox_width - text_width) / 2  # Center the line
        draw.text((x, y), line, font=font, fill=gradient_color)  # Use gradient color

        y += text_bbox[3] - text_bbox[1] + 5  # Move y position down for the next line

    return image
