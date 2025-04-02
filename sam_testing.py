from ultralytics import SAM
import torch

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load a model and move it to GPU
model = SAM("sam2.1_b.pt").to(device)

# Display model information (optional)
model.info()

# Run inference with GPU and save output as video
model("testing.mp4", save=True, device=device)