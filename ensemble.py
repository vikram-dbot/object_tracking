import torch

def to_tensor(x):
    """Convert x to a tensor if possible; if it's not numeric, return None."""
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, (int, float)):
        return torch.tensor(x)
    elif isinstance(x, (list, tuple)):
        try:
            return torch.tensor(x)
        except Exception as e:
            return None
    else:
        # For non-numeric types (e.g. str, dict), return None
        return None

def average_model_weights(state_dict1, state_dict2, alpha=0.5):
    averaged_state_dict = {}
    for key in state_dict1:
        if key in state_dict2:
            tensor1 = to_tensor(state_dict1[key])
            tensor2 = to_tensor(state_dict2[key])
            if (tensor1 is not None) and (tensor2 is not None):
                # Both values can be converted to tensors; average them.
                averaged_state_dict[key] = alpha * tensor1 + (1 - alpha) * tensor2
            else:
                # For keys that aren't numeric (or cannot be converted), just keep the first value.
                averaged_state_dict[key] = state_dict1[key]
        else:
            raise ValueError(f"Key '{key}' not found in both state dictionaries.")
    return averaged_state_dict

# Example usage:
model1_path = r"yolo_averaged_model.pt"  # Path to your first fine-tuned model.
model2_path = r"C:\Users\Welcome\phases_detection\runs\detect\train3\weights\best.pt"  # Path to your second model.

# Load state dictionaries (adjust map_location if needed)
state_dict1 = torch.load(model1_path, map_location="cpu")
state_dict2 = torch.load(model2_path, map_location="cpu")

# If your checkpoint files store additional metadata (e.g., under a key "model"), extract the model params:
# state_dict1 = state_dict1["model"]
# state_dict2 = state_dict2["model"]

averaged_state_dict = average_model_weights(state_dict1, state_dict2, alpha=0.5)

# Save the averaged state dictionary
averaged_model_path = "yolo_averaged_model_3.pt"
torch.save(averaged_state_dict, averaged_model_path)

print(f"Averaged model weights saved to {averaged_model_path}")
