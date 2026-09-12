import torch
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from transformers import CLIPProcessor, CLIPModel
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.cuda.empty_cache()

images_dir = "./trainingData/images/"

# Initiliase SAM
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

# Initialise CLIP
clip_model_id = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(clip_model_id)
clip_model = CLIPModel.from_pretrained(clip_model_id).to(device)

# Create class-agnostic masks with SAM
# Load image (HWC format in RGB)
filename = "video1_0000.jpg"
image_path = os.path.join(images_dir, filename)
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Generate masks
masks = mask_generator.generate(image)
print (f"Generated {len(masks)} masks")