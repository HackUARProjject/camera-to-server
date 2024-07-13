from flask import Flask, request, jsonify  # Flask関連モジュールのインポート
import cv2  # OpenCVライブラリのインポート
import numpy as np  # NumPyライブラリのインポート

app = Flask(__name__)  # Flaskアプリケーションのインスタンスを作成

# SSD MobileNetの設定
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")  # モデル構造と学習済み重みの読み込み

# クラス名の読み込み（SSD MobileNet用）
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]  # クラス名をリストに読み込み

@app.route('/detect', methods=['POST'])  # /detectエンドポイントを定義し、POSTメソッドでアクセス可能に
def detect():
    file = request.files['image']  # リクエストから画像ファイルを取得
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)  # 画像をデコードしてNumPy配列に変換

    result = detect_objects(img)  # 物体検出関数を呼び出して結果を取得

    return jsonify(result)  # 物体検出結果をJSON形式で返す

def detect_objects(img):
    height, width = img.shape[:2]  # 画像の高さと幅を取得
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)  # 画像を前処理してSSD MobileNet用のブロブを作成
    net.setInput(blob)  # 前処理した画像をモデルに入力
    detections = net.forward()  # モデルで前方伝播し、検出結果を取得

    result = []  # 検出結果を格納するリスト
    for i in range(detections.shape[2]):  # 各検出結果に対して
        confidence = detections[0, 0, i, 2]  # 信頼度を取得
        if confidence > 0.5:  # 信頼度が0.5以上の場合
            class_id = int(detections[0, 0, i, 1])  # クラスIDを取得
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])  # バウンディングボックスの座標を元の画像サイズにスケーリング
            (x, y, x2, y2) = box.astype("int")  # バウンディングボックスの座標を整数に変換
            label = str(classes[class_id])  # クラスIDからラベル名を取得
            result.append({
                'label': label,  # ラベル名
                'confidence': float(confidence),  # 信頼度
                'bbox': [x, y, x2, y2]  # バウンディングボックスの座標
            })

    return result  # 検出結果を返す

if __name__ == '__main__':  # このスクリプトがメインプログラムとして実行される場合
    app.run(host='0.0.0.0', port=5000)  # サーバーを0.0.0.0:5000で起動し、外部からのアクセスを受け付ける
