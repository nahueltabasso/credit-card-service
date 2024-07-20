from abc import ABC, abstractmethod
import cv2
import numpy as np

class ObjectDetector(ABC):
    @abstractmethod
    def set_target_image(self, image_path):
        pass
    @abstractmethod
    def load_reference_images(self, image_paths):
        pass
    @abstractmethod
    def detect(self):
        pass

class SIFTObjectDetector(ObjectDetector):
    def __init__(self, match_threshold=20, lowe_ratio=0.7):
        self.sift = cv2.SIFT_create()
        self.target_image = None
        self.target_kp = None
        self.target_des = None
        self.reference_images = {}
        self.match_threshold = match_threshold
        self.lowe_ratio = lowe_ratio

    def preprocess_image(self, image):
        # Convert to grayscale if it's not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        # Apply histogram equalization
        # equalized = cv2.equalizeHist(gray)
        # # Apply Gaussian blur
        # blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        # return blurred
        return gray

    def set_target_image(self, image):
        self.target_image = self.preprocess_image(image)
        self.target_kp, self.target_des = self.sift.detectAndCompute(self.target_image, None)

    def load_reference_images(self, image_paths):
        for name, path in image_paths.items():
            image = cv2.imread(path)
            processed_image = self.preprocess_image(image)
            kp, des = self.sift.detectAndCompute(processed_image, None)
            self.reference_images[name] = {
                'image': processed_image, 
                'kp': kp, 
                'des': des, 
                'original': image
            }

    def detect(self):
        if self.target_image is None or not self.reference_images:
            raise ValueError("Target image and reference images must be loaded before detection")
        
        best_match = None
        max_good_matches = 0
        
        for name, ref in self.reference_images.items():
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(ref['des'], self.target_des, k=2)
            
            good_matches = []
            for m, n in matches:
                if m.distance < self.lowe_ratio * n.distance:
                    good_matches.append(m)
            
            print(f"{name}: {len(good_matches)} good matches")
            
            if len(good_matches) > max_good_matches:
                max_good_matches = len(good_matches)
                best_match = name

        # Verificar si el mejor match supera el umbral
        if max_good_matches >= self.match_threshold:
            return best_match
        return None

    def check_color(self, image, colors, threshold=1000):
        for color in colors:
            mask = cv2.inRange(image, np.array(color) - 30, np.array(color) + 30)
            if cv2.countNonZero(mask) > threshold:
                return True
        return False