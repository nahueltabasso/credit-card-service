from groundingdino.util.inference import Model
import numpy as np
import torch

MODEL_CONFIG_PATH = "../config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "../../weights/groundingdino_swint_ogc.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

def predict(img: np.ndarray):
    """Performs credit card detection in an image using the GroundingDINO
    model.
    
    This function loads the pre-trained GroundingDINO model and uses it to
    detect credit cards in the image provided. Use a text prompot to guide
    detection.

    Args:
        img (np.ndarray): image of a credit card in numpy
    array format. It is expected to be a color image (BGR or RGB)

    Returns:
        _type_: return and Detections object. Generally include
            - 'xyxy': coors of the bounding boxes detected
            - 'scores': confidence for each detection
    
    Raises:
        ValueError: if the input image is not valid or is empty
        RuntimeError: if there are problems to load the model or
    make the inference
    
    Notes:
        - Uses a pre-trained GroundingDINO model specific in MODEL_CONFIG
        - The detections make with thresholds pre-defined
        - The execution runtime (CPU/CUDA) is defined on automatic way
        - The prompt used is "credit card"
    """
    model = Model(model_config_path=MODEL_CONFIG_PATH,
              model_checkpoint_path=WEIGHTS_PATH,
              device=device)
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25
    TEXT_PROMPT = ["credit card"]

    detections = model.predict_with_classes(image=img,
                                        classes=TEXT_PROMPT,
                                        box_threshold=BOX_THRESHOLD,
                                        text_threshold=TEXT_THRESHOLD)
    print(f"GroundingDino detections --- {detections}")
    return detections