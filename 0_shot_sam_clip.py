import torch
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from transformers import CLIPProcessor, CLIPModel
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.cuda.empty_cache()

images_dir = "./trainingData/images/"

# Initialise SAM
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

# Initialise CLIP
clip_model_id = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(clip_model_id)
clip_model = CLIPModel.from_pretrained(clip_model_id).to(device)

def segment(filename):
    # Create class-agnostic masks with SAM
    # Load image (HWC format in RGB)
    image_path = os.path.join(images_dir, filename)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Generate masks
    masks = mask_generator.generate(image)
    print (f"Generated {len(masks)} masks")

    # Extract image patches (ROIs)
    image_pil  = Image.fromarray(image)
    cropped_patches = []

    for mask in masks:
        # SAM gives bounding boxes for masks as [x,y,w,h]
        x, y, w, h = mask['bbox']
        # Crop the image using the bounding box
        cropped = image_pil.crop((x, y, x+w, y+h))
        cropped_patches.append(cropped)

    return cropped_patches, masks

def label_segments(cropped_patches, masks):
    # Text prompts for CLIP
    classes = [
        "clear blue sky",
        "cloudy sky",
        "ground",
        "water",
        "mountains",
        "trees",
        "buildings"
    ]
    text_prompts = [f"a photo of a {label}" for label in classes]

    # Pre-tokenise the text prompts
    #? Why?
    text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True).to(device)

    # Classify each SAM patch using CLIP
    pseudo_labels = []
    confidence_threshold = 0.85
    for patch, mask_data in zip(cropped_patches, masks):
        # Process the image patch
        image_inputs = processor(images=patch, return_tensors="pt").to(device)

        with torch.no_grad(): #? What does no_grad do?
            # Get similarities
            outputs = clip_model(**image_inputs, **text_inputs) #? What does ** do?
            logits_per_image = outputs.logits_per_image # image-text similarity score
            probs = logits_per_image.softmax(dim=1) # normalise to probabilities

        # Get the top prediction
        max_prob, top_idx = torch.max(probs, dim=1)

        if max_prob.item() > confidence_threshold:
            predicted_class = classes[top_idx.item()]

            # Save the result
            pseudo_labels.append({
                "segmentation": mask_data["segmentation"], # the boolean mask
                "bbox" : mask_data["bbox"], # the bounding box
                "class_label" : predicted_class,
                "confidence" : max_prob.item()
            })


