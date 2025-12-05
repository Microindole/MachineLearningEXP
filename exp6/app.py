import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for

# ==========================================
# 1. 配置与模型定义 (必须与训练时一致)
# ==========================================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 检测设备
device = torch.device("cpu") # 部署时通常用 CPU 即可，防止显存冲突

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 7), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# ==========================================
# 2. 加载训练好的模型
# ==========================================
print("正在加载模型...")
try:
    # 加载自编码器
    ae_model = Autoencoder().to(device)
    ae_model.load_state_dict(torch.load('results/autoencoder.pth', map_location=device))
    ae_model.eval()

    # 加载 CNN
    cnn_model = CNN(num_classes=3).to(device)
    cnn_model.load_state_dict(torch.load('results/cnn.pth', map_location=device))
    cnn_model.eval()
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败，请检查 results 文件夹下是否有 .pth 文件。\n错误信息: {e}")
    exit()

# 类别名称 (顺序必须与训练时 dataset.class_to_idx 一致)
# 训练时的顺序通常是字母序: Covid, Normal, Viral Pneumonia
CLASS_NAMES = ['Covid (新冠)', 'Normal (正常)', 'Viral Pneumonia (病毒性肺炎)']

# 预处理 (必须与训练集保持一致：灰度 -> Resize 64x64)
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# ==========================================
# 3. Web 路由逻辑
# ==========================================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            # 1. 保存图片 (建议存到 static 目录以便浏览器访问)
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 2. 推理
            try:
                image = Image.open(filepath)
                img_tensor = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    denoised_img = ae_model(img_tensor)
                    output = cnn_model(denoised_img)
                    probs = torch.softmax(output, dim=1)[0] # 获取概率分布

                    # 3. 整理所有类别的概率数据
                    results = []
                    for i, name in enumerate(CLASS_NAMES):
                        score = probs[i].item() * 100
                        results.append({
                            'name': name,
                            'score': f"{score:.2f}",
                            'value': score
                        })

                    # 按概率从高到低排序
                    results.sort(key=lambda x: x['value'], reverse=True)

                    # 最佳结果
                    top_prediction = results[0]

                return render_template('index.html',
                                       prediction=top_prediction,
                                       all_results=results,
                                       image_url=filepath)
            except Exception as e:
                print(f"Error: {e}")
                return render_template('index.html', error="图片处理失败，请重试")

    return render_template('index.html', prediction=None)

# 用于显示上传的图片
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename))

if __name__ == '__main__':
    print("系统启动中... 请在浏览器访问 http://127.0.0.1:5000")
    app.run(debug=True)