from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
import pandas as np

app = Flask(__name__)

# YOLOv3モデルをロード
model = torch.hub.load('ultralytics/yolov3', 'custom', path='best.pt')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # 物体検出を実行
    result = detect_objects(img)

    return jsonify(result)

def detect_objects(img):
    # 画像をYOLOv3モデルに入力
    results = model(img)

    # 検出結果をパース
    results = results.xyxy[0].cpu().numpy()
    result = []
    for *bbox, conf, cls in results:
        x_min, y_min, x_max, y_max = map(int, bbox)
        label = model.names[int(cls)]
        result.append({
            'label': label,
            'confidence': float(conf),
            'bbox': [x_min, y_min, x_max, y_max]
        })
    return result

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
