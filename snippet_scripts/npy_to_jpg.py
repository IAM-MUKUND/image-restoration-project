import numpy as np
from PIL import Image

def npy_to_jpeg(npy_path, output_jpg_path):
    # Load raw numpy array (float32)
    arr = np.load(npy_path)
    
    # Scale from [0.0, 1.0] to [0, 255] and clip out-of-bound values caused by noise
    img_u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    
    # Convert uint8 2D array into PIL Image
    img = Image.fromarray(img_u8, mode='L')
    img.save(output_jpg_path, 'JPEG', quality=95)
