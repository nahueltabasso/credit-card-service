from src.models.model import CreditCardData
from src.utils.file_utils import extract_zone, preprocess_img
from datetime import datetime
from typing import List, Tuple, Dict
import easyocr
import numpy as np
import re

class OCRService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OCRService, cls).__new__(cls)
            cls._instance.reader = easyocr.Reader(['en'])
            cls._instance.img = None
            cls._instance.elements = None
            cls._instance.zones = None
        return cls._instance

    def set_img(self, img: np.ndarray) -> None:
        self.img = preprocess_img(image=img)

    def set_images(self, elements: dict) -> None:
        self.elements = elements

    def set_zones_coords(self, zones: Dict[str, Tuple[int, int, int, int]]) -> None:
        self.zones = zones

    def _extract_text(self, zone_name: str) -> List[Tuple]:
        if self.img is None or self.zones is None or self.elements == {}:
            raise ValueError("Image and zones and elements must be set before extraction")
        if self.elements[zone_name] is None:
            zone = extract_zone(img=self.img, zone=self.zones[zone_name])
        else:
            zone = self.elements[zone_name]
        return self.reader.readtext(zone)

    def _format_text(self, results: List[Tuple], formatter: callable) -> str:
        return formatter([text for _, text, _ in results])

    def get_credit_card_number(self, img: np.ndarray, zone: Tuple[int, int, int, int]) -> str:
        number_zone = extract_zone(img=preprocess_img(image=img), zone=zone)
        results = self.reader.readtext(number_zone)
        return self._format_text(results, self._format_card_number)

    def extract(self, entity: CreditCardData) -> CreditCardData:
        extractions = {
            'card_number': (self._extract_text('card_number'), self._format_card_number),
            'cardholder': (self._extract_text('cardholder'), self._format_card_name),
            'expiry_date': (self._extract_text('expiry_date'), self._format_expiration_date)
        }
        
        print(f"Credit card number after OCR - {extractions['card_number']}")
        print(f"Name after OCR - {extractions['cardholder']}")
        print(f"Expiration adte after OCR - {extractions['expiry_date']}")

        for attr, (results, formatter) in extractions.items():
            setattr(entity, attr, self._format_text(results, formatter))

        entity.create_at = datetime.now()
        return entity

    @staticmethod
    def _format_card_number(texts: List[str]) -> str:
        return ''.join(texts)

    @staticmethod
    def _format_card_name(texts: List[str]) -> str:
        return ' '.join(texts).upper()

    @staticmethod
    def _format_expiration_date(texts: List[str]) -> str:
        pattern = r'^[0-9/]+$'
        return ''.join(text for text in texts if re.match(pattern, text))

def get_ocr_service():
    return OCRService()
