from src.core.gd_inference import predict
from src.core.object_detector import SIFTObjectDetector
from src.core.ocr_service import get_ocr_service
from src.utils.file_utils import crop_image, show_image, extract_zone
from src.config.config import Config
from src.models.model import CreditCardData
from dotenv import load_dotenv
import numpy as np
import cv2 
import supervision as sv
import traceback
import os
import requests 

load_dotenv("../../.env")
config = Config()

def credit_card_detector(img: np.ndarray, show: bool=False):
    """Detect and analyse a credit card on an image
    
    This function make the follows steps:
    1. Detect a credit card on an image
    2. Extract the zone of the credit card
    3. Identify the Payment Network

    Args:
        img (np.ndarray): input image in numpy array format
        show (bool, optional): if is true, show the input image and
    cropped image. Default value is false
    
    Returns;
        Tuple[np.ndarray, str]: a tuple that include:
            - np.ndarray: cropped image or None if not detected any credit card
            - str: name of the Payment Network, None if can not define
    """
    detections = predict(img=img)    
    bbox = get_credit_card_bbox(detections=detections)
    print(f"Bounding Box of Credit Card --- {bbox}")
    if bbox is not None:
        credit_card = crop_image(img=img,
                                 bbox=bbox)

        if show:
            show_image(img, "Input Image")
            show_image(credit_card, "Credit Card")
            
        payment_network = identify_payment_network(credit_card=credit_card)
        print(f"Payment Network --- {payment_network}")
        return credit_card, payment_network
    return None, None
    
def get_credit_card_bbox(detections: sv.Detections) -> list:
    """This method returns a bounding box of credit card from 
    detections object if inside that object there are 1 detection else
    return None

    Args:
        detections (sv.Detections): _description_

    Returns:
        _type_: _description_
    """
    xyxy = detections.xyxy
    
    if len(xyxy) == 1:
        xyxy = [int(coord) for coord in xyxy[0]]
        print(f"TYPE XYXY ---- {type(xyxy)}")
    else:
        xyxy = None
    return xyxy    
    
def identify_payment_network(credit_card: np.ndarray) -> str:
    """This method identifies the Payment Network of a credit card
    from its image.
    
    This feature uses a two-step approach to identify the 
    Payment Network:
    
    1. Try to extract the numbers from the card throuhg OCR and
    use the IIN (Issuer Identification Number) to defined a 
    Payment Network
    2. If step 1 fails, resort to logo detection using SIFT Algorithm

    Args:
        credit_card (np.ndarray): image of a credit card in numpy
    array format. It is expected to be a color image (BGR or RGB)

    Returns:
        str: name of the Payment Network identified (e.g, "VISA", "MASTERCARD")
    """
    # TODO: develop a identify by IIN logic
    payment_network = identify_by_IIN(credit_card=credit_card)
    if payment_network is not None:
        return payment_network
    return identify_by_SIFT(credit_card=credit_card)

def identify_by_IIN(credit_card: np.ndarray) -> str:
    """Identify the Payment Network of a credit card using it IIN
    
    This function make the follows steps
    1. Extract the credit card number by OCR technique
    2. Try to identified the payment network using a extern service (BINLIST)
    3. If failed the extern service, try to identify the payment network locally 

    Args:
        credit_card (np.ndarray): credit card image in a numpy array format

    Returns:
        str: the name of the payment network. Return None if can not identify it
        
    Notes: 
        - This method assumes that the necessary configuration (such as 
    COMMON_CARD_NUMBER_ZONE) is defined in the config module.
        - The method uses helper functions (get_ocr_service, get_payment_networt, etc)
    that must be defined in the scope
    """
    ocrService = get_ocr_service()
    zone = config.COMMON_CARD_NUMBER_ZONE
    card_number = ocrService.get_credit_card_number(img=credit_card,
                                                    zone=zone)
    
    if len(card_number) > 6:
        # First try to identify the payment network by BINLIST
        payment_network = get_payment_network(card_number=card_number)
        
        if payment_network is None:
            # Try to identify the payment network by a locally way
            payment_network = get_payment_network_local(card_number=card_number)
            return payment_network
        
        return payment_network
    return None

def identify_by_SIFT(credit_card: np.ndarray) -> str:
    """Identifies the Payment Network of a credit card using SIFTH Algorithm
    
    This function extract the zone or ROI where it is common to find the logo
    of Payment Network. Then apply the SIFT algorithm to detect and compare
    characteristics with references images and determine the Payment Network
    based on best match

    Args:
        credit_card (np.ndarray): image of a credit card in numpy
    array format. It is expected to be a color image (BGR or RGB)

    Returns:
        str: name of the Payment Network identified (e.g, "VISA", "MASTERCARD")
    """
    # Extract payment network zone
    zone = extract_zone(img=credit_card,
                        zone=config.COMMON_PAYMENT_NETWORK_ZONE)
    
    # Create a new instance of SIFT Detector
    detector = SIFTObjectDetector(match_threshold=20, lowe_ratio=0.7)
    # Set a target image to detector
    detector.set_target_image(image=zone)
    # Load reference images
    detector.load_reference_images(config.PATTERNS_DICT)
    # Make the matching
    result = detector.detect()
    
    if result != None:
        if result.startswith("visa"):
            type = config.VISA_CONSTANT
        elif result.startswith("american_"):
            type = config.AMERCIAN_EXPRESS_CONSTANT
        elif result.startswith("cabal"):
            type = config.CABAL_CONSTANT
        elif result.startswith("mastercard"):
            type = config.MASTERCARD_CONSTANT
    else:
        type = None
        
    return type

def get_zones_coords(payment_network: str) -> dict:
    if payment_network == "VISA":
        return config.VISA_CREDIT_CARD_ZONES
    elif payment_network == "AMERICAN EXPRESS":
        return config.AMERICAN_CREDIT_CARD_ZONES
    elif payment_network == "MASTERCARD":
        return config.MASTER_CREDIT_CARD_ZONES
    else:
        return config.CABAL_CREDIT_CARD_ZONES

def get_payment_network(card_number: str) -> str:
    """Define the payment network by BINLIST service

    Args:
        card_number (str): full credit card number

    Raises:
        ValueError: if the url of BINLIST API is not in our environments variables

    Returns:
        str | None: the name of the payment network or None if can not determine the 
    network or appear some error in the request
    """
    api_url = os.getenv('BINLIST_API_URL')
    
    if not api_url:
        raise ValueError("The API url is not set up in your environments variables")
    
    # BINLIST only accept the first 6 digits
    iin = card_number[:6]
    
    endpoint = f"{api_url}/{iin}"
    try:
        response = requests.get(endpoint)
        response.raise_for_status()
        data = response.json()
        
        payment_network = data.get("scheme")
        if payment_network.startswith("visa"):
            return config.VISA_CONSTANT
        elif payment_network.startswith("mastercard"):
            return config.MASTERCARD_CONSTANT
        elif payment_network.startswith("american"):
            return config.AMERCIAN_EXPRESS_CONSTANT
        else:
            return config.CABAL_CONSTANT
    except requests.RequestException as e:
        print(f"Error in the request to API: {e}")
        traceback.print_exc()
        return None

def get_payment_network_local(card_number: str) -> str:
    """ Determines the payment network based on the first digits of the
    credit card number.
    
    The rules are:
        - Visa: Start with 4
        - Mastercard: Start with 51, 52, 53, 54, o 55
        - American Express: Start with 34 o 37
        - Cabal: Start with 6

    Returns:
        str | None: the name of the payment network or None if can not identify it
    """
    if card_number.startswith("4"):
        return config.VISA_CONSTANT
    if card_number.startswith(("51", "52", "53", "54", "55")):
        return config.MASTERCARD_CONSTANT
    if card_number.startswith(("34", "37")):
        return config.AMERCIAN_EXPRESS_CONSTANT
    if card_number.startswith("6"):
        return config.CABAL_CONSTANT
    return None
   
