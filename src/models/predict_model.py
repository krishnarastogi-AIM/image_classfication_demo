import os
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from simple_cnn import SimpleCNN
import neptune
from PIL import Image
import argparse
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
NEPTUNE_WORKSPACE_PROJECT = os.getenv("YOUR_WORKSPACE/YOUR_PROJECT_NAME")
NEPTUNE_API_TOKEN = os.getenv("YOUR_API_TOKEN")

# Initialize Neptune
neptune.init(NEPTUNE_WORKSPACE_PROJECT, api_token=NEPTUNE_API_TOKEN)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('../../models/cifar10_simple_cnn.pth'))
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def predict(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    output = model(image)
    prob = F.softmax(output, dim=1)
    prediction = output.argmax(dim=1, keepdim=True).item()
    confidence = prob[0][prediction].item()

    # Log the prediction and confidence to Neptune
    neptune.create_experiment(name='cifar10_prediction')
    neptune.log_metric('prediction', prediction)
    neptune.log_metric('confidence', confidence)
    neptune.stop()

    return prediction, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict the class of an image using the trained CIFAR-10 model.')
    parser.add_argument('image_path', type=str, help='Path to the input image.')
    args = parser.parse_args()
    
    prediction, confidence = predict(args.image_path)
    print(f"Predicted class: {datasets.CIFAR10.classes[prediction]}, Confidence: {confidence:.4f}")

## to run ```python predict_model.py path_to_your_image.jpg```