import torch
from torchviz import make_dot
from RetinexFormer_arch import RetinexFormer

# Load the model architecture
model = RetinexFormer(n_feat=32)
checkpoint = torch.load("/home/ancuti/expendables/Retinexformer/experiments/ISALux - 27.38/best_psnr_27.38_37440.pth")
print(checkpoint.keys())
model_state_dict = checkpoint['params']


# Load the model weights from the .pth file
model.load_state_dict(model_state_dict)
# Set the model to evaluation mode (this is important if you are not training)
model.eval()

dummy_input = torch.randn(8, 3, 256, 256)

# Pass the dummy input through the model
output = model(dummy_input)

# Generate a visualization using torchviz
dot = make_dot(output, params=dict(model.named_parameters()))

# Save the graph to a file
dot.render("transformer_model_from_pth", format="png")  # This will save as 'transformer_model_from_pth.png'
