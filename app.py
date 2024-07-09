# 各ライブラリのインポート
import streamlit as st
from langchain_community.llms import LlamaCpp

# モデルの準備、プロンプト編集する部分
def use_model(text):
    # 使用するモデルの名前
    MODEL_NAME = ".model/Llama-3-ELYZA-JP-8B-q4_k_m.gguf"
    # モデルの準備
    llm = LlamaCpp(
        model_path=MODEL_NAME,
    )
    # プロンプト編集部分
    prompt = "あなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、常に日本語で回答してください。{}。".format(text)
    # モデルの実行
    return llm.invoke(prompt)
# UI(チャットのやり取りの部分)
def main():
    # タイトル
    st.title("LLMアプリ")
    # 入力フォームと送信ボタンのUI
    st.chat_message("assistant").markdown("何か聞きたいことはありませんか？")
    text = st.chat_input("ここにメッセージを入力してください")

    # チャットのUI
    if text:        
        st.chat_message("user").markdown(text)
        st.chat_message("assistant").markdown(use_model(text))

if __name__ == "__main__":
    main()
