import easyocr
import numpy as np
from PIL import Image
from typing import Union
from utils.helpers import get_logger, time_it

logger = get_logger("ocr")

class OCRService:
    """Service to lazily initialize EasyOCR and run in-memory text extraction on document page images."""

    def __init__(self):
        self._reader = None

    def get_reader(self) -> easyocr.Reader:
        """Lazily initializes the EasyOCR Reader instance (only once)."""
        if self._reader is None:
            try:
                logger.info("Initializing EasyOCR reader with language: ['en']...")
                # Download models dynamically on first execution. Runs on CPU by default.
                self._reader = easyocr.Reader(['en'], gpu=False)
                logger.info("EasyOCR reader initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR reader: {e}")
                raise RuntimeError(f"EasyOCR Init Failure: {e}")
        return self._reader

    @time_it
    def extract_text_from_image(self, image: Union[Image.Image, np.ndarray]) -> str:
        """
        Accepts a PIL Image or NumPy array, runs OCR, and returns the concatenated string of extracted text.
        Handles OCR failures gracefully by logging the error and returning an empty string.
        
        Args:
            image: A PIL Image or numpy ndarray.
            
        Returns:
            Extracted text string, or empty string on failure.
        """
        try:
            reader = self.get_reader()
            
            # Convert PIL image to numpy array if necessary, since easyocr works best with paths, bytes, or ndarrays.
            if isinstance(image, Image.Image):
                # Convert PIL to NumPy
                img_arr = np.array(image)
            else:
                img_arr = image

            logger.info("Running EasyOCR on page image...")
            # readtext returns a list of tuples: (bbox, text, confidence)
            results = reader.readtext(img_arr)
            
            if not results:
                logger.warning("EasyOCR run yielded empty text output.")
                return ""

            # Concatenate all extracted text segments with spaces
            extracted_text = " ".join([segment[1] for segment in results if segment[1]])
            confidence_scores = [segment[2] for segment in results if segment[2]]
            
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            logger.info(f"EasyOCR finished extraction. Extracted {len(results)} segments. Average Confidence: {avg_confidence:.2f}")
            
            return extracted_text.strip()
            
        except Exception as e:
            logger.error(f"Failed to extract text from image using EasyOCR: {e}")
            # Do not crash the application, return an empty string to allow graceful native fallback
            return ""

# Create a singleton instance for global reusability
ocr_service = OCRService()

def extract_text_from_image(image: Union[Image.Image, np.ndarray]) -> str:
    """Reusable wrapper around the global OCRService singleton."""
    return ocr_service.extract_text_from_image(image)
