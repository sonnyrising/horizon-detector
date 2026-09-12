import os
import torch
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

images_dir = "./trainingData/images"
output_dir = "./trainingData/lab_masks"
SKY_CLASS_ID = 2

os.makedirs(output_dir, exist_ok=True)

# Load the model
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

# Move the operations to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing on: {device}")
model.to(device)

print("Starting")
for filename in os.listdir(images_dir):
    # Skip non-image files
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    print(f"Processing {filename}...")
    image_path = os.path.join(images_dir, filename)
    image = Image.open(image_path).convert("LAB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    outputs = model(**inputs)
    logits = outputs.logits

    upsampled_logits = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    predicted_mask = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
    binary_mask = (predicted_mask == SKY_CLASS_ID).astype(np.uint8) * 255

    mask_image = Image.fromarray(binary_mask)
    save_path = os.path.join(output_dir, filename.replace('.jpg', '.png'))
    mask_image.save(save_path)

print("All images processed!")