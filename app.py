from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import timm
import pennylane as qml
import sqlite3
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import warnings
warnings.filterwarnings('ignore')
from werkzeug.utils import secure_filename
import cv2  # OpenCV is already imported for face detection

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / 'static'
UPLOAD_DIR = STATIC_DIR / 'uploads'
DB_PATH = BASE_DIR / 'database.db'

app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

CLASS_NAMES = ['angry','disgust','fear','happy','neutral','sad','surprise']
EMOTION_EMOJI = {
    'angry':   '😠',
    'disgust': '🤢',
    'fear':    '😨',
    'happy':   '😊',
    'neutral': '😐',
    'sad':     '😢',
    'surprise':'😲'
}
NUM_CLASSES = 7
N_QUBITS    = 4
DEVICE      = torch.device('cpu')

MODEL_DIR      = BASE_DIR / 'models'
GOOGLENET_PATH = MODEL_DIR / 'GoogleNet.pth'
RESNET101_PATH = MODEL_DIR / 'ResNet101-TL.pth'
SWIN_TL_PATH   = MODEL_DIR / 'swin_tl_100epochs.pth'
MODEL_F_PATH   = MODEL_DIR / 'model_f.pth'

def ensure_upload_dir():
    """Ensure upload directory exists, creating it if necessary."""
    try:
        if UPLOAD_DIR.exists():
            if not UPLOAD_DIR.is_dir():
                print(f"⚠️ Conflict: {UPLOAD_DIR} exists as a file. Removing it.")
                UPLOAD_DIR.unlink()  # Remove the file
        os.makedirs(str(UPLOAD_DIR), exist_ok=True)
        print(f"✅ Upload directory ensured: {UPLOAD_DIR}")
    except Exception as e:
        print(f"❌ Error ensuring upload directory: {e}")
        raise

# ── Database ───────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        image_name  TEXT,
        predicted   TEXT,
        confidence  REAL,
        angry       REAL,
        disgust     REAL,
        fear        REAL,
        happy       REAL,
        neutral     REAL,
        sad         REAL,
        surprise    REAL,
        timestamp   TEXT
    )''')
    conn.commit()
    conn.close()

def save_prediction(image_name, predicted, confidence, probs):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO predictions
        (image_name, predicted, confidence, angry, disgust, fear,
         happy, neutral, sad, surprise, timestamp)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
        (image_name, predicted, confidence,
         float(probs[0]), float(probs[1]), float(probs[2]),
         float(probs[3]), float(probs[4]), float(probs[5]),
         float(probs[6]),
         datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM predictions ORDER BY id DESC LIMIT 50')
    rows = c.fetchall()
    conn.close()
    return rows

def get_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT predicted, COUNT(*) FROM predictions GROUP BY predicted')
    stats = dict(c.fetchall())
    c.execute('SELECT COUNT(*) FROM predictions')
    total = c.fetchone()[0]
    conn.close()
    return stats, total

# ── Model Architectures ────────────────────────────────────────
class InceptionBlock(nn.Module):
    def __init__(self, in_ch, ch1, ch3r, ch3, ch5r, ch5, pool):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, ch1, 1), nn.BatchNorm2d(ch1), nn.ReLU())
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, ch3r, 1), nn.BatchNorm2d(ch3r), nn.ReLU(),
            nn.Conv2d(ch3r, ch3, 3, padding=1), nn.BatchNorm2d(ch3), nn.ReLU())
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, ch5r, 1), nn.BatchNorm2d(ch5r), nn.ReLU(),
            nn.Conv2d(ch5r, ch5, 5, padding=2), nn.BatchNorm2d(ch5), nn.ReLU())
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_ch, pool, 1), nn.BatchNorm2d(pool), nn.ReLU())
    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x),
                          self.branch3(x), self.branch4(x)], 1)

class GoogleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 192, 3, padding=1),
            nn.BatchNorm2d(192), nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1))
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.pool3  = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.pool4  = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(512, NUM_CLASSES)
    def forward(self, x):
        x = self.stem(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.pool4(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

class ResNet101Transfer(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet101(pretrained=False)
        backbone.fc = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, NUM_CLASSES))
        self.model = backbone
    def forward(self, x):
        return self.model(x)

class SwinTL(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224', pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Linear(768, 512), nn.GELU(), nn.BatchNorm1d(512), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.GELU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES))
    def forward(self, x):
        return self.classifier(self.backbone(x))

DEV2 = qml.device('lightning.qubit', wires=N_QUBITS)
@qml.qnode(DEV2, interface='torch', diff_method='adjoint')
def circuit_f(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

class SwinQuantumExact(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = timm.create_model(
            'swin_tiny_patch4_window7_224', pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.GELU(), nn.BatchNorm1d(256), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, N_QUBITS))
        self.qlayer = qml.qnn.TorchLayer(circuit_f, {"weights": (2, N_QUBITS, 3)})
        self.output = nn.Linear(N_QUBITS, NUM_CLASSES)
    def forward(self, x):
        x = self.swin(x)
        x = torch.tanh(self.classifier(x)) * np.pi
        x = self.qlayer(x)
        return self.output(x)

# ── Load Models ────────────────────────────────────────────────
print("Loading models... please wait ~2 minutes")
googlenet = GoogleNet().to(DEVICE)
googlenet.load_state_dict(torch.load(GOOGLENET_PATH, map_location=DEVICE))
googlenet.eval()
print("✅ GoogleNet loaded!")

resnet101 = ResNet101Transfer().to(DEVICE)
resnet101.load_state_dict(torch.load(RESNET101_PATH, map_location=DEVICE))
resnet101.eval()
print("✅ ResNet101-TL loaded!")

swin_tl = SwinTL().to(DEVICE)
ckpt = torch.load(SWIN_TL_PATH, map_location=DEVICE)
swin_tl.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
swin_tl.eval()
print("✅ Swin-TL loaded!")

model_f = SwinQuantumExact().to(DEVICE)
ckpt_f  = torch.load(MODEL_F_PATH, map_location=DEVICE)
model_f.load_state_dict(ckpt_f['model_state_dict'] if 'model_state_dict' in ckpt_f else ckpt_f)
model_f.eval()
print("✅ Swin+VQC loaded!")
print("🎉 All models ready! Starting server...")

# ── Transform ──────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ── Prediction ─────────────────────────────────────────────────
def predict(image):
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out1 = torch.softmax(swin_tl(img_tensor),  dim=1)
        out2 = torch.softmax(resnet101(img_tensor), dim=1)
        out3 = torch.softmax(googlenet(img_tensor), dim=1)
        out4 = torch.softmax(model_f(img_tensor),  dim=1)
    ensemble   = 0.40*out1 + 0.25*out2 + 0.20*out3 + 0.15*out4
    probs      = ensemble[0].cpu().numpy()
    pred_idx   = int(np.argmax(probs))
    emotion    = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx] * 100)
    return emotion, confidence, probs

# ── Helper function for face detection
def detect_faces(image_path):
    """Detect faces in an image using OpenCV."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(str(image_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0  # Return True if faces are detected, otherwise False

# ── Routes ─────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', emojis=EMOTION_EMOJI)


@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        ensure_upload_dir()

        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Unsupported file type'}), 400

        secure_name = secure_filename(file.filename)
        if not secure_name:
            return jsonify({'error': 'Invalid file name'}), 400

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{uuid4().hex[:8]}_{secure_name}"
        save_path = UPLOAD_DIR / filename

        try:
            file.save(save_path)
        except Exception as e:
            return jsonify({'error': f'Failed to save file: {str(e)}'}), 500

        if not save_path.exists():
            return jsonify({'error': 'File was not saved successfully'}), 500

        # Detect faces in the uploaded image
        if not detect_faces(save_path):
            save_path.unlink()  # Remove the invalid file
            return jsonify({'error': 'Invalid photo: No human face detected'}), 400

        try:
            image = Image.open(save_path).convert('RGB')
        except Exception:
            if save_path.exists():
                save_path.unlink()
            return jsonify({'error': 'Uploaded file is not a valid image'}), 400

        emotion, confidence, probs = predict(image)
        save_prediction(filename, emotion, confidence, probs)

        return jsonify({
            'emotion': emotion,
            'emoji': EMOTION_EMOJI[emotion],
            'confidence': round(confidence, 2),
            'probs': {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2) for i in range(NUM_CLASSES)},
            'image_url': f'/static/uploads/{filename}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def history():
    rows = get_history()
    stats, total = get_stats()
    return render_template('history.html',
                           rows=rows, stats=stats,
                           total=total, emojis=EMOTION_EMOJI)

@app.route('/api/stats')
def api_stats():
    stats, total = get_stats()
    return jsonify({'stats': stats, 'total': total})

@app.route('/camera', methods=['GET', 'POST'])
def camera():
    if request.method == 'GET':
        return render_template('camera.html')

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({'error': 'Unable to access the camera'}), 500

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({'error': 'Failed to capture image from camera'}), 500

    # Save the captured frame temporarily
    temp_path = UPLOAD_DIR / 'temp_camera_capture.jpg'
    cv2.imwrite(str(temp_path), frame)

    # Detect faces in the captured frame
    if not detect_faces(temp_path):
        temp_path.unlink()  # Remove the invalid file
        return jsonify({'error': 'No human face detected in the captured image'}), 400

    # Process the image for emotion detection
    try:
        image = Image.open(temp_path).convert('RGB')
        emotion, confidence, probs = predict(image)
        temp_path.unlink()  # Remove the temporary file after processing

        return jsonify({
            'emotion': emotion,
            'emoji': EMOTION_EMOJI[emotion],
            'confidence': round(confidence, 2),
            'probs': {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2) for i in range(NUM_CLASSES)}
        })
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    ensure_upload_dir()
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
