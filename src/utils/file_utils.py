from fastapi import UploadFile, HTTPException
from PIL import Image
import numpy as np
import cv2
import io

def crop_image(img: np.ndarray, bbox: tuple) -> np.ndarray:
    """This methods crops an image according to its
    bounding box 

    Args:
        img (np.ndarray): _description_
        bbox (tuple): _description_

    Returns:
        np.ndarray: _description_
    """
    return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

def show_image(img: np.ndarray, label:str) -> None:
    """Show an image with OpenCV

    Args:
        img (np.ndarray): _description_
        label (str): _description_
    """
    cv2.imshow(label, img)
    cv2.waitKey()

def to_fixed(img: np.ndarray, bbox: tuple) -> tuple:
    """This method convert relatives coords from a bbox to
    absolutes coords

    Args:
        img (np.ndarray): _description_
        bbox (tuple): _description_

    Returns:
        tuple: _description_
    """
    # h, w, _ = img.shape
    if len(img.shape) == 3:
        h, w, _ = img.shape
    elif len(img.shape) == 2:
        h, w = img.shape

    relative_bbox = [ (int(b[0]*w), int(b[1]*h)) for b in bbox ]

    return relative_bbox 
    
def extract_zone(img: np.ndarray, zone: list) -> np.ndarray:
    """Cuts out a zone (bbox) and returns it

    Args:
        img (np.ndarray): opencv image
        zone (list): bbox in format [(x1, y1), (x2, y2)]

    Returns:
        np.ndarray: zone cutted from the image
    """

    p1, p2 = to_fixed(img, zone)
    return img[p1[1]:p2[1], p1[0]:p2[0], ...]

def preprocess_img(image: np.ndarray, save: bool = False) -> np.ndarray:
    """This function make a pre-process on image: convert to gray scale and 
    increase the contrast and brightness.
    
    Make the follows steps to its pre-process:
    1. Verify that input image be valid
    2. Convert input image to a gray scale
    3. Increase image contrast
    4. Save images to visual inspection

    Args:
        image (np.ndarray): Input image in numpy array format.

    Raises:
        FileNotFoundError: _description_

    Returns:
        _type_: output image in numpy array format
    """
    if image is None:
        raise FileNotFoundError("Imagen no valida")
    
    # Convert img to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if save:
        cv2.imwrite("./gray.jpg", gray)
    # Increase constrast and brightness
    contrast = 1.5
    bright = 0
    enhanced = cv2.addWeighted(gray, contrast, gray, 0, bright)

    # Make sure the values are in the range [0, 255]
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    if save:
        cv2.imwrite("./enhanced.jpg", enhanced)
    return enhanced

def validate_image(file: UploadFile):
    try:
        # Try to open file as an image
        image = Image.open(io.BytesIO(file.file.read()))
        return image
    except Exception:
        raise HTTPException(status_code=400,
                            detail="File not valid!")
        
def image_to_numpy(image: Image.Image) -> np.ndarray:
    return np.array(image)
    