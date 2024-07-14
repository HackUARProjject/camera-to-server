import openai

openai.api_key = 'ここにキーをいれてね'  # ここにAPIキーを直接設定

# サンプルラベル名
label = "person"

def get_descriptions(label):
    descriptions = {}
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"オブジェクト '{label}' を日本語で20文字以内で詳細に説明してください。"}
            ],
            max_tokens=50
        )
        descriptions[label] = response['choices'][0]['message']['content'].strip()
    except openai.error.RateLimitError:
        descriptions[label] = "レート制限を超えました。しばらくしてからもう一度お試しください。"
    except openai.error.InvalidRequestError as e:
        descriptions[label] = f"無効なリクエスト: {e}"
    except Exception as e:
        descriptions[label] = f"エラーが発生しました: {e}"
    return descriptions

# ラベル名の説明文を取得
descriptions = get_descriptions(label)
print(descriptions)
