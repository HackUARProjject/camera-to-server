import requests  # HTTPリクエストを送るためのライブラリをインポート
import openai
import cv2
import numpy as np
from datetime import datetime
from PIL import ImageFont, ImageDraw, Image

labels = ""

# OpenAI APIキーの設定　あんまり使わないでね
openai.api_key = 'sk-None-OAfrPPG4AMl5r1boMDKWT3BlbkFJf4VEdzaoYukXHlQvZUHw'

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

def main():
    frame = capture_frame()  # フレームをキャプチャ

    if frame is not None:
        send_frame_to_server(frame)  # フレームをサーバーに送信


if __name__ == "__main__":
    main()
    # 画像を読み込む
    image_path = "black.png"
    label = labels[0]['label']
    descriptions = get_descriptions(label)
    # print(descriptions)
    description_text = descriptions.get(label, "")
    image = cv2.imread(image_path)

    # バウンディングボックスの色と太さを定義
    bbox_color = (0, 255, 0)  
    bbox_thickness = 25

    # バウンディングボックスを描画
    height, width, _ = image.shape
    cv2.rectangle(image, (0, 0), (width, height - 80), bbox_color, bbox_thickness)  # 高さを少し減らして下にスペースを作る
    

    # 現在の時刻を取得する
    current_time = datetime.now().strftime('%H:%M')

    # 時刻のテキストのフォント、サイズ、色を定義
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 4
    font_color = bbox_color 
    font_thickness = 5

    # 画像に時刻のテキストを描画
    cv2.putText(image, current_time, (30, 100), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    # 右下のアクションボタンの表示
    bottom_text = "アクションボタンを押して検索を開始"

    #アクションボタンの位置を調整(x座標, y座標)
    bottom_text_position = (1400, image.shape[0] - 60) 


    #日本語に対応させる
    image = putText_japanese(image, bottom_text, bottom_text_position, 30, (0, 255, 0))

    #画面文字の改行文字数を変えたい場合ここを変える
    char_num = 10

    if len(description_text) % char_num == 0:
        for i in range(len(description_text) // char_num):
            description_position = (1350, 300 + 300 * (0.25 * i))
            image = putText_japanese(image, description_text[i * char_num:char_num * (i + 1)], description_position, 50, (0, 255, 0))
    else:
        for i in range((len(description_text) // char_num) + 1):
            description_position = (1350, 300 + 300 * (0.25 * i))
            image = putText_japanese(image, description_text[i * char_num:char_num * (i + 1)], description_position, 50, (0, 255, 0))


    # ウィンドウサイズを設定して画像を表示
    cv2.namedWindow('Image with Bounding Box, Time, and Description', cv2.WINDOW_NORMAL)
    # ウィンドウサイズを800x600に設定
    cv2.resizeWindow('Image with Bounding Box, Time, and Description', 1400, 1200)
    cv2.imshow('Image with Bounding Box, Time, and Description', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
