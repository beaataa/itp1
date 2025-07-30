import os
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from ultralytics import YOLO

# ----- Config -----
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'best_segmentation_model.pth'
NUM_CLASSES = 8
IMG_SIZE = (480, 640)  # H, W

# ----- Flask setup -----
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----- Load model -----
# Use default weights to initialize full structure (incl. aux_classifier)
model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
yolo_model = YOLO('yolov8.pt')

# Replace classifier output layers to match your training
model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
if model.aux_classifier is not None:
    model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)

# Load saved weights
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# ----- Image transforms -----
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----- Segmentation logic -----
def predict_mask(image):
    tensor = transform(image).unsqueeze(0)  # [1, 3, H, W]
    with torch.no_grad():
        output = model(tensor)['out']
        mask = torch.argmax(output.squeeze(), dim=0).byte().cpu().numpy()
    return mask

def yolo_predict_mask(image):
    import cv2
    
    image_resized = image.resize((640, 480))
    results = yolo_model.predict(image_resized, save=False, conf=0.03)[0]

    width, height = image.size
    final_mask = np.zeros((height, width), dtype=np.uint8)

    if results.masks:
        masks = results.masks.data.cpu().numpy()  # Shape: [num_instances, H, W]
        classes = results.boxes.cls.cpu().numpy().astype(np.uint8)  # Class IDs per instance

        for mask_array, cls_id in zip(masks, classes):
            mask_resized = cv2.resize(mask_array, (width, height), interpolation=cv2.INTER_NEAREST)
            final_mask[mask_resized > 0.5] = cls_id + 1  

    return final_mask

def save_colored_mask(mask, original_size, save_path):
    palette = np.array([
        [0, 0, 0],         # 0 background
        [128, 0, 0],       # 1 bolt
        [0, 128, 0],       # 2 busbar
        [128, 128, 0],     # 3 cable
        [0, 0, 128],       # 4 connector
        [128, 0, 128],     # 5 nut
        [0, 128, 128],     # 6 plasticfilm
        [128, 128, 128]    # 7 plasticcover
    ], dtype=np.uint8)
    color_mask = palette[mask]
    mask_img = Image.fromarray(color_mask)
    mask_img = mask_img.resize(original_size)
    mask_img.save(save_path)

# ----- Routes -----
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']    
    filename = file.filename
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)

    image = Image.open(input_path).convert('RGB')
    
    deeplab_mask = predict_mask(image)
    yolo_mask = yolo_predict_mask(image)
    
    deeplab_filename = f"mask_deeplab_{filename}"
    yolo_filename = f"mask_yolo_{filename}"

    deeplab_path = os.path.join(UPLOAD_FOLDER, deeplab_filename)
    yolo_path = os.path.join(UPLOAD_FOLDER, yolo_filename)

    save_colored_mask(deeplab_mask, image.size, deeplab_path)
    save_colored_mask(yolo_mask, image.size, yolo_path)

    return jsonify({
        'original': '/' + input_path.replace('\\', '/'),
        'deeplab': '/' + deeplab_path.replace('\\', '/'),
        'yolo': '/' + yolo_path.replace('\\', '/')
    })

# ----- Run app -----
if __name__ == '__main__':
    app.run(debug=True)