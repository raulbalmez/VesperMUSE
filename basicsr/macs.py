import torch
from torchprofile import profile_macs
from basicsr.models.archs.VesperMUSE_arch import VesperMUSE_General_Seg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VesperMUSE_General_Seg(n_feat=32).to(device)
input_tensor = torch.randn(1, 4, 256, 256).to(device)

# --- Compute MACs/FLOPs ---
macs = profile_macs(model, input_tensor)
flops = macs * 2
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
tflops = flops / (1024 * 1024 * 1024)

print("=" * 70)
print(f"🔹 Model Total Params: {num_params/1e6:.3f} M")
print(f"🔹 Model MACs: {macs / (1024*1024*1024):.3f} G")
print(f"🔹 Model FLOPs: {tflops:.3f} G")
print("=" * 70)

# --- Moderate recursive breakdown ---
def summarize_module(module, name="", indent=0, depth=2):
    """Recursively summarize module parameters up to a given depth."""
    total_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    prefix = " " * indent
    print(f"{prefix}{name or module.__class__.__name__:<30s}: {total_params/1e6:.3f} M params")

    if depth > 0:
        for child_name, child in module.named_children():
            # Skip leaf ops like Conv2d, ReLU, etc.
            if len(list(child.children())) == 0:
                continue
            summarize_module(child, child_name, indent + 4, depth - 1)

print("\n🔍 Module-level parameter breakdown:\n")
summarize_module(model, "VesperMUSE_General_Seg", depth=3)
print("=" * 70)
