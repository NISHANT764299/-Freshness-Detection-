from flask import Flask, request, render_template
import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import os

class FreshnessCNN(nn.Module):
    def __init__(self, num_classes=20):
        super(FreshnessCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

app = Flask(__name__)


model = FreshnessCNN(num_classes=20)
model.load_state_dict(torch.load('freshness_cnn.pth', map_location=torch.device('cpu')))
model.eval()


model.classes = [
    'Apple_Fresh', 'Apple_Stale', 'Apple_Rotten',
    'Potato_Fresh', 'Potato_Stale', 'Potato_Rotten',
    'Tomato_Fresh', 'Tomato_Stale', 'Tomato_Rotten',
    'Banana_Fresh', 'Banana_Stale', 'Banana_Rotten',
    'Carrot_Fresh', 'Carrot_Stale', 'Carrot_Rotten',
    'Onion_Fresh', 'Onion_Stale', 'Onion_Rotten',
    'Cabbage_Fresh', 'Cabbage_Stale'
]

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return "No file uploaded"

    file = request.files['file']
    img = Image.open(file.stream)
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted_class = torch.max(outputs, 1)

    predicted_label = model.classes[predicted_class.item()]
    
    return render_template('result.html', label=predicted_label)


if __name__ == '__main__':
    app.run(debug=True)
