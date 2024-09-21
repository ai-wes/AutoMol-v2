import numpy as np
from skimage import filters, measure, morphology
import logging

class TelomeraseLocalization:
    def __init__(self):
        logging.info("Initialized TelomeraseLocalization.")

    def analyze_telomerase_localization(self, image_data: np.ndarray) -> float:
        try:
            logging.info("Performing image analysis for telomerase localization...")
            blurred = filters.gaussian(image_data, sigma=1)
            thresh = filters.threshold_otsu(blurred)
            binary = blurred > thresh
            cleaned = morphology.remove_small_objects(binary, min_size=30)
            labeled_img = measure.label(cleaned)
            props = measure.regionprops(labeled_img, intensity_image=image_data)
            localization_score = np.mean([prop.mean_intensity for prop in props]) if props else 0
            logging.info(f"Telomerase Localization Score: {localization_score}")
            return localization_score
        except Exception as e:
            logging.error(f"Error in analyze_telomerase_localization: {e}")
            raise
