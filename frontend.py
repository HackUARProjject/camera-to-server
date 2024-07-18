import requests  # HTTPリクエストを送るためのライブラリをインポート
import openai
import cv2
import numpy as np
import keyboard
import time
import asyncio
from datetime import datetime
from PIL import ImageFont, ImageDraw, Image


labels = ""

# OpenAI APIキーの設定　あんまり使わないでね
openai.api_key = ''

def capture_frame():
    cap = cv2.VideoCapture(0)  # カメラのIDが0の場合、カメラを開く
    if not cap.isOpened():
        print("カメラを開けません")  # カメラが開けなかった場合のエラーメッセージ
        return None  # フレームを取得できないのでNoneを返す

    ret, frame = cap.read()  # カメラからフレームを取得
    cap.release()  # カメラを解放

    if not ret:
        print("フレームを取得できませんでした")  # フレームが取得できなかった場合のエラーメッセージ
        return None  # フレームを取得できないのでNoneを返す

    return frame  # 取得したフレームを返す

def frame_to_jpeg(frame):
    ret, jpeg = cv2.imencode('.jpg', frame)  # フレームをJPEG形式にエンコード
    if not ret:
        print("フレームのエンコードに失敗しました")  # エンコードが失敗した場合のエラーメッセージ
        return None  # エンコードに失敗した場合はNoneを返す
    return jpeg.tobytes()  # エンコードされた画像データをバイト形式で返す

def send_frame_to_server(frame):
    url = "http://127.0.0.1:5000/detect"  # サーバーのURLを指定
    jpeg_data = frame_to_jpeg(frame)  # フレームをJPEG形式に変換
    if jpeg_data is None:
        return  # 変換に失敗した場合は処理を中止

    files = {'image': ('frame.jpg', jpeg_data, 'image/jpeg')}  # 送信するファイルを指定
    response = requests.post(url, files=files)  # POSTリクエストを送信

    if response.status_code == 200: 
        global labels
        labels = response.json()
        print("サーバーからの応答:", response.json())  # サーバーからの応答を表示
    else:
        print("サーバーへのリクエストが失敗しました")  # リクエストが失敗した場合のエラーメッセージ

#説明文を取得する関数
def get_descriptions(label):
    descriptions = {}
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"オブジェクト '{label}' を日本語で、小学生の子供でもわかるように40文字以内で説明してください。"}
            ],
            max_tokens=100
        )
        descriptions[label] = response['choices'][0]['message']['content'].strip()
    except openai.error.RateLimitError:
        descriptions[label] = "レート制限を超えました。しばらくしてからもう一度お試しください。"
    except openai.error.InvalidRequestError as e:
        descriptions[label] = f"無効なリクエスト: {e}"
    except Exception as e:
        descriptions[label] = f"エラーが発生しました: {e}"
    return descriptions

#cv2が日本語対応ではないので、日本語にするための処理
def putText_japanese(img, text, point, size, color):
    #Raspberry Pi OSで実行する場合'/usr/share/fonts/opentype/NotoSansCJK-Bold.ttc'からパスを指定してください
    #windows11でNotoSansCJK-BOLD.ttcを使用する場合こちらから"https://github.com/notofonts/noto-cjk/blob/main/Sans/OTC/NotoSansCJK-Bold.ttc"ダウンロードしてください(--;)
    font = ImageFont.truetype('', size)
    
    #imgをndarrayからPILに変換
    img_pil = Image.fromarray(img)
    
    #drawインスタンス生成
    draw = ImageDraw.Draw(img_pil)
    
    #テキスト描画
    draw.text(point, text, fill=color, font=font)
    
    #PILからndarrayに変換して返す
    return np.array(img_pil)

def view_result_img():
    image_path = "black.png"
    image = cv2.imread(image_path)
    
    if not labels:
        text = "物体が検出されませんでした"
        image = putText_japanese(image, text, (image.shape[1]//2 - 300, image.shape[0]//2), 50, (0, 255, 0))
    else:
        label = labels[0]['label']
        descriptions = get_descriptions(label)
        description_text = descriptions.get(label, "")

        bbox_color = (0, 255, 0)
        bbox_thickness = 25
        height, width, _ = image.shape
        cv2.rectangle(image, (0, 0), (width, height - 80), bbox_color, bbox_thickness)

        char_num = 10
        lines = [description_text[i:i+char_num] for i in range(0, len(description_text), char_num)]
        for i, line in enumerate(lines):
            description_position = (1350, 300 + 300 * (0.25 * i))
            image = putText_japanese(image, line, description_position, 50, (0, 255, 0))

    current_time = datetime.now().strftime('%H:%M')
    cv2.putText(image, current_time, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 5, cv2.LINE_AA)
    
    cv2.imshow('image', image)
    cv2.waitKey(5000)  # 5秒間表示

def idle_screen():
    image_path = "black.png"
    image = cv2.imread(image_path)

    current_time = datetime.now().strftime('%H:%M')

    # 時刻のテキストを描画
    cv2.putText(image, current_time, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 5, cv2.LINE_AA)

    # アクションボタンの表示
    bbox_color = (0, 255, 0)  
    bbox_thickness = 25
    height, width, _ = image.shape
    cv2.rectangle(image, (0, 0), (width, height - 80), bbox_color, bbox_thickness) 
    # アクションボタンの表示
    bottom_text = "1キーを押して検索を開始"
    bottom_text_position = (1400, image.shape[0] - 60)
    image = putText_japanese(image, bottom_text, bottom_text_position, 30, (0, 255, 0))

    return image


def main():
    frame = capture_frame()  # フレームをキャプチャ

    if frame is not None:
        send_frame_to_server(frame)  # フレームをサーバーに送信
        print("done")

def countdown(num):
    for i in range(num, -1, -1):
        image_path = "black.png"
        image = cv2.imread(image_path)
        
        if i == 0:
            text = "しばらくお待ちください"
            image = putText_japanese(image, text, (image.shape[1]//2 - 300, image.shape[0]//2), 50, (0, 255, 0))
        else:
            text = str(i)
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 7, 10)[0]
            text_x = (image.shape[1] - textsize[0]) // 2
            text_y = (image.shape[0] + textsize[1]) // 2
            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 255, 0), 10, cv2.LINE_AA)

        cv2.imshow('image', image)
        cv2.waitKey(1000)
        

def main_loop():
    cv2.namedWindow('image', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        image = idle_screen()
        cv2.imshow('image', image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'): 
            countdown(3)
            main() # 既存のmain関数を呼び出し
            view_result_img()  # 結果を表示
            time.sleep(5)  # 結果表示後、5秒待機

        elif key == 27:  # ESCキーでプログラム終了
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()