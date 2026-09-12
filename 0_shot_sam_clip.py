import torch
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from torch._dynamo.testing import np
from transformers import CLIPProcessor, CLIPModel
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.cuda.empty_cache()

images_dir = "./trainingData/images"
output_dir = "./trainingData/SAM_CLIP_masks"
os.makedirs(output_dir, exist_ok=True)


# Initialise SAM
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

# Initialise CLIP
clip_model_id = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(clip_model_id)
clip_model = CLIPModel.from_pretrained(clip_model_id).to(device)

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



def segment(filename):
    # Create class-agnostic masks with SAM
    # Load image (HWC format in RGB)
    image_path = os.path.join(images_dir, filename)
    image = cv2.imread(image_path)

    # Downscale the image by 50% to save VRAM
    height, width = image.shape[:2]
    image = cv2.resize(image, (width // 2, height // 2))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Generate masks
    with torch.no_grad():
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

    return image, cropped_patches, masks

def label_segments(cropped_patches, masks):
    # Pre-tokenise the text prompts
    #? Why?
    text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True).to(device)

    # Classify each SAM patch using CLIP
    pseudo_labels = []
    confidence_threshold = 0.30
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

    print(f"Kept {len(pseudo_labels)} out of {len(masks)} masks after CLIP threshold.")
    return pseudo_labels


def generate_image(image, pseudo_labels, filename):
    h, w = image.shape[:2]

    # Initialize with 2 (Unclassified/Unknown) instead of 0
    semantic_mask = np.full((h, w), 2, dtype=np.uint8)

    class_to_id = {
        "clear blue sky": 0,
        "cloudy sky": 0,
        "ground": 1,
        "water": 1,
        "mountains": 1,
        "trees": 1,
        "buildings": 1
    }

    pseudo_labels.sort(key=lambda x: x["segmentation"].sum(), reverse=True)

    for label_data in pseudo_labels:
        bool_mask = label_data["segmentation"]
        predicted_class = label_data["class_label"]

        # Default to 2 if somehow not found
        class_id = class_to_id.get(predicted_class, 2)
        semantic_mask[bool_mask] = class_id

    # For PyTorch training, you might want Unclassified to be 0 or 1 later,
    # but for now, save the raw integer mask
    cv2.imwrite(os.path.join(output_dir, f"mask_{filename}"), semantic_mask)

    # --- CREATE A 3-COLOR DEBUG MASK ---
    debug_mask = np.zeros((h, w), dtype=np.uint8)  # Start Black
    debug_mask[semantic_mask == 0] = 128  # Sky = Gray
    debug_mask[semantic_mask == 1] = 255  # Ground = White
    # (Unclassified remains 0 = Black)

    cv2.imwrite(os.path.join(output_dir, f"mask_debug_{filename}"), debug_mask)


for filename in os.listdir(images_dir):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    print(f"Processing {filename}...")

    # Catch the image returned by segment()
    image, cropped_patches, masks = segment(filename)

    pseudo_labels = label_segments(cropped_patches, masks)

    # Pass both the image and filename to generate_image()
    generate_image(image, pseudo_labels, filename)




