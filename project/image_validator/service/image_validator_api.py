import torch
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from torchvision import transforms
from PIL import Image, ImageStat
from PIL.ImageStat import Stat
from io import BytesIO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

log_file = "background_removal.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Global scope
weights = RetinaNet_ResNet50_FPN_Weights.COCO_V1
model = retinanet_resnet50_fpn(weights=weights)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
transform = transforms.Compose([transforms.ToTensor()])


# Function to check if an image is blank or solid color
def is_blank_or_solid_color(image_data, threshold=5):
    try:
        image = Image.open(BytesIO(image_data)).convert("RGB")
        stat = ImageStat.Stat(image)
        stddev = stat.stddev
        return all(s < threshold for s in stddev)
    except Exception as e:
        logger.error(f"Error checking blank or solid color: {str(e)}")
        return False

# Function to check if an image has an unusual aspect ratio
def is_unusual_aspect_ratio(image_data, min_ratio=0.5, max_ratio=2.0):
    try:
        image = Image.open(BytesIO(image_data))
        width, height = image.size
        ratio = width / height
        return ratio < min_ratio or ratio > max_ratio
    except Exception as e:
        logger.error(f"Error checking aspect ratio: {str(e)}")
        return False

# Function to check if an image is too small
def is_too_small(image_data, min_width=300, min_height=300):
    try:
        image = Image.open(BytesIO(image_data))
        width, height = image.size
        return width < min_width or height < min_height
    except Exception as e:
        logger.error(f"Error checking image size: {str(e)}")
        return False
    
# Function to check if an image has low entropy (indicating low information content)
def is_low_entropy(image_data, threshold=3.0):
    try:
        image = Image.open(BytesIO(image_data)).convert("L")
        entropy = image.entropy()
        return entropy < threshold
    except Exception as e:
        logger.error(f"Error checking entropy: {str(e)}")
        return False
    
# Function to detect objects in an image using a pre-trained model
def detect_objects_uploaded(image_data):
    """
    Function to detect objects in an image using a pre-trained RetinaNet model.
    Returns True if objects are detected, False otherwise.
    """
    try:
        # Load model with updated weights argument
        weights = RetinaNet_ResNet50_FPN_Weights.COCO_V1
        model = retinanet_resnet50_fpn(weights=weights)
        model.eval()

        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Open image and preprocess
        image = Image.open(BytesIO(image_data)).convert("RGB")
        input_tensor = preprocess(image)
        input_batch = [input_tensor]

        # Move model and inputs to the same device (cuda or cpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        input_batch = [tensor.to(device) for tensor in input_batch]

        with torch.no_grad():
            predictions = model(input_batch)

        scores = predictions[0]["scores"].cpu().numpy()
        valid_indices = scores > 0.1  # Lowered threshold to 0.1 for more sensitivity

        # Log the scores for debugging
        logger.info(f"Detection scores: {scores}")
        
        # Return whether there are any valid detections
        return valid_indices.any()
    except Exception as e:
        logger.error(f"Error in object detection: {str(e)}")
        return False

def validate_image_for_upload(image_data):
    if is_blank_or_solid_color(image_data):
        logger.info("Image rejected: Blank or solid color")
        return False, "Image is blank or solid color"

    if is_unusual_aspect_ratio(image_data):
        logger.info("Image rejected: Unusual aspect ratio")
        return False, "Unusual aspect ratio"

    if is_too_small(image_data):
        logger.info("Image rejected: Image too small")
        return False, "Image too small"

    if not detect_objects_uploaded(image_data):
        logger.warning("Object detection failed; trying entropy check as fallback")
        if is_low_entropy(image_data):
            logger.info("Image rejected: No object detected and low entropy")
            return False, "No object detected in image"
        else:
            logger.info("Image passed fallback entropy check")

    logger.info("Image passed validation")
    return True, "Image is valid"