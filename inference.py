import torch
from model.model import FrozenInTime
import json

# Load configuration from JSON file
with open('configs/eval/epic.json', 'r') as f:
    config = json.load(f)

# initialize the model
model = FrozenInTime(**config['arch']['args'])

# Get the path of the checkpoint
checkpoint_path = config['arch']['args']['load_checkpoint']
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Prepare the data of inference
# Example input tensor (batch_size=1, channels=3, frames=16, height=224, width=224)
input_data = torch.randn(1, 3, 16, 224, 224)  

with torch.no_grad():
    # Perform inference
    output = model(input_data)
    print("Inference output:", output)