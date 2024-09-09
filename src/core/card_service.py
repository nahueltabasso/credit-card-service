from ultralytics import YOLO
from dotenv import load_dotenv
from src.utils.file_utils import crop_image, show_image
from src.config.config import Config
import numpy as np
import os
import torchvision as tv


class CardService:
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            # Load environment variables
            script_dir = os.path.dirname(os.path.abspath(__file__))
            env_path = os.path.join(script_dir, '../../.env')

            if not os.path.exists(env_path):
                raise FileNotFoundError(f".env file not found in path: {env_path}")

            load_dotenv(env_path)
            
            print("Environment variables loaded from ", env_path)
            card_detector_model = os.getenv('YOLO_CARD_DETECTOR')
            elements_detector_model = os.getenv('YOLO_CARD_ELEMENT_DETECTOR')
            classifier_model = os.getenv('YOLO_PAYMENT_NETWORK_CLASSIFIER')
            
            # Verificar que las variables de entorno estén correctamente cargadas
            if not all([card_detector_model, elements_detector_model, classifier_model]):
                raise EnvironmentError("Failed in load environment variables")
            
            print(f"YOLO_CARD_DETECTOR: {card_detector_model}")
            print(f"YOLO_CARD_ELEMENT_DETECTOR: {elements_detector_model}")
            print(f"YOLO_PAYMENT_NETWORK_CLASSIFIER: {classifier_model}")
            
            cls._instance = super(CardService, cls).__new__(cls)
            cls.card_detector = YOLO(model=card_detector_model)
            cls.elements_detector = YOLO(model=elements_detector_model)
            cls.classifier = YOLO(model=classifier_model)
                    
        return cls._instance
    
    def _inference(self, detector: any, img: np.ndarray, iou_threshold: float=0.5) -> tuple:
        result = detector(source=img)

        xyxy = result[0].boxes.xyxy.cpu()
        clss = result[0].boxes.cls.cpu()
        confs = result[0].boxes.conf.cpu()

        # Aplicar NMS (Non-Max Suppression)
        filtered_id_boxes = tv.ops.nms(boxes=xyxy, scores=confs, iou_threshold=iou_threshold)

        # Seleccionar y convertir los cuadros de delimitación filtrados a una lista
        xyxy = xyxy[filtered_id_boxes].tolist()
        # Seleccionar y convertir las etiquetas de clase filtradas a una lista
        clss = clss[filtered_id_boxes].tolist()

        return xyxy, clss

    
    def get_card_bbox(self, input_img: np.ndarray, show: bool=False):
        result = self.card_detector(source=input_img)
        
        # Validate if the model detected only one credit/debit card, else the image is not valid
        xyxy, clss = self._inference(detector=self.card_detector,
                                      img=input_img)
            
        if len(xyxy) == 1:
            # Cast to int the coordinates
            xyxy = [int(xy) for xy in xyxy[0]]
            print(f"Credit card box -> {xyxy} - Class -> {clss[0]}")
            credit_card = crop_image(img=input_img, bbox=xyxy)
            
            if show:
                show_image(img=input_img, label="Input Image")
                show_image(img=credit_card, label="Output Image")
                
            return credit_card
        
        return None
    
    def get_card_elements(self, card: np.ndarray, show: bool=False) -> dict:
        xyxy, clss = self._inference(detector=self.elements_detector,
                                     img=card)
        
        # Cut the card elements from credit card image and set dictionary with values
        elements_dict = self._set_elements(card=card,
                                      boxes=xyxy,
                                      clss=clss)
        if show:
            [show_image(img=value, label=key) for key, value in elements_dict.items()]
        return elements_dict
    
    def classify_payment_network(self, element: np.ndarray, card: np.ndarray) -> str:
        config = Config()
        if element is None:
            element = crop_image(img=card,
                                 bbox=config.COMMON_CARD_ZONES['payment_network'])
        result = self.classifier(source=element)
        top1 = result[0].probs.top1
        top1_conf = result[0].probs.top1conf
        
        if top1 == 0:
            return config.AMERCIAN_EXPRESS_CONSTANT
        elif top1 == 1:
            return config.CABAL_CONSTANT
        elif top1 == 2:
            return config.MASTERCARD_CONSTANT
        elif top1 == 3:
            return config.VISA_CONSTANT
        else:
            return None
        
    
    @staticmethod
    def _set_elements(card: np.ndarray, boxes: list, clss: list) -> dict:
        dict = {}
        card_number = None
        cardholder = None
        expiry_date = None
        payment_network = None
        for box, cls in zip(boxes, clss):
            box = [int(b) for b in box]
            print(f"Box {box} - cls {cls}")
            if cls == 0.0:
                card_number = crop_image(img=card, bbox=box)
            elif cls == 1.0:
                expiry_date = crop_image(img=card, bbox=box)
            elif cls == 2.0:
                cardholder = crop_image(img=card, bbox=box)
            elif cls == 3.0:
                payment_network = crop_image(img=card, bbox=box)
            else:
                pass
        
        dict['card_number'] = card_number if card_number is not None else None
        dict['expiry_date'] = expiry_date if expiry_date is not None else None
        dict['cardholder'] = cardholder if cardholder is not None else None
        dict['payment_network'] = payment_network if payment_network is not None else None
        return dict

    
def get_card_service():
    return CardService()