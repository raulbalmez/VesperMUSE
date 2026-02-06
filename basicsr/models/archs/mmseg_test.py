import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

# Initialize the model and feature extractor
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

# Prepare your tensor (3, H, W) with appropriate size and normalize it to [0, 1]
input_tensor = torch.randn(1, 3, 256, 256)  # Replace with your actual tensor
input_tensor = torch.clamp(input_tensor, 0, 1)  # Ensure values are within [0, 1]

# Normalize and prepare input
inputs = feature_extractor(images=input_tensor, return_tensors="pt").pixel_values
with torch.no_grad():
    outputs = model(inputs)

# Access the logits directly as a tensor
logits = outputs.logits  # Shape: [batch_size, num_classes, height, width]
print(logits.shape)
