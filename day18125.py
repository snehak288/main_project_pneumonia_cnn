from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from PIL import Image
import os

app = Flask(__name__)

# Load the saved model
def load_model(model_path, device):
    from torchvision import models
    import torch.nn as nn

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Define the transformation for the input image
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to predict the class of an image
def predict_image(image_path, model, device):
    img = Image.open(image_path).convert('RGB')
    img = image_transforms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img).squeeze(1)
        probability = torch.sigmoid(output).item()

    if probability >= 0.5:
        return "Pneumonia", probability
    else:
        return "Normal", probability

# Routes
@app.route('/')
def main():
    return render_template('main.html')  # Render the main page

@app.route('/upload')
def upload():
    return render_template('uploaaad.html')  # Render the upload page

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    image_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(image_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "pneumonia_resnet18_model.pth"

    if not os.path.exists(model_path):
        return jsonify({'error': f"Model file '{model_path}' not found"}), 500

    model = load_model(model_path, device)
    label, confidence = predict_image(image_path, model, device)

    os.remove(image_path)

    return jsonify({'label': label, 'confidence': round(confidence, 4)})

if __name__ == '__main__':
    app.run(debug=True)
