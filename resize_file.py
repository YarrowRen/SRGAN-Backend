from PIL import Image
import os


def resize_and_save_image(image_path):
    # Load the image
    image = Image.open(image_path)

    # Calculate the new size: half of the original size
    new_size = (image.width // 2, image.height // 2)

    # Resize the image using LANCZOS resampling method
    resized_image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Construct the new file name
    base, ext = os.path.splitext(image_path)
    new_image_path = f"{base}_half{ext}"

    # Save the resized image
    resized_image.save(new_image_path)

    # Return the new image path
    return new_image_path



