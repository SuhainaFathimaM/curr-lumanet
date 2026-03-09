import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# --------------------------------------------------
# Model Definition (MUST MATCH TRAINING NOTEBOOK)
# --------------------------------------------------
class LumaNet(nn.Module):
    def __init__(self):
        super(LumaNet, self).__init__()
        number_f = 16
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.conv7(torch.cat([x1, x6], 1)))

        r = torch.split(x_r, 3, dim=1)

        for i in range(8):
            x = x + r[i] * (torch.pow(x, 2) - x)

        return x


# --------------------------------------------------
# Flask App Setup
# --------------------------------------------------
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Use GPU if available (better than forcing CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LumaNet().to(device)

# --------------------------------------------------
# Load Weights (Correct Way for statdict)
# --------------------------------------------------
weight_path = "weights.pth"

if os.path.exists(weight_path):
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    print("✅ Weights loaded successfully")
else:
    print("❌ weights.pth not found!")


# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        if 'file' not in request.files:
            return "No file uploaded"

        file = request.files['file']
        if file.filename == '':
            return "No selected file"

        fname = secure_filename(file.filename)

        in_p = os.path.join(UPLOAD_FOLDER, fname)
        out_p = os.path.join(RESULT_FOLDER, "out_" + fname)

        file.save(in_p)

        # -----------------------------
        # Image Processing
        # -----------------------------
        img = cv2.imread(in_p)

        if img is None:
            return "Error reading image"

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_t = torch.from_numpy(img_rgb).float() / 255.0
        img_t = img_t.permute(2, 0, 1).unsqueeze(0).to(device)

        # -----------------------------
        # Inference
        # -----------------------------
        with torch.no_grad():
            result = model(img_t)

        result = result.squeeze().permute(1, 2, 0).cpu().numpy()
        result = np.clip(result, 0, 1)

        result_bgr = cv2.cvtColor((result * 255).astype(np.uint8),
                                  cv2.COLOR_RGB2BGR)

        cv2.imwrite(out_p, result_bgr)

        return render_template('index.html',
                               original_img=in_p,
                               enhanced_img=out_p)

    return render_template('index.html')


# --------------------------------------------------
# Run App
# --------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)