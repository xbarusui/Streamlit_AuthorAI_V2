# -*- coding: utf-8 -*-

import streamlit as st
import ai_learning as learning
import ai_learning_novelup as learning_n
import ai_generate as generate
import text_analyze as analyze
import uploadfile as upload

def main():

    # タイトル
    st.title('創作作家AI')

    # アプリケーション名と対応する関数のマッピング
    apps = {
        "-": None,
        "AIテキスト学習": ai_learning,
        "AIテキスト学習(ノベプラ)": ai_learning_novelup,
        "AIモデルアップロード": ai_uploadfile,
        "AIテキスト生成": ai_generate,
        "テキスト可視化": text_wordcloud,
        "テキスト分析テスト": text_analyze,
        "テキスト要約テスト": text_summarize
    }
    selected_app_name = st.sidebar.selectbox(label="apps",
                                             options=list(apps.keys()))

    if selected_app_name == "-":
        st.info("Please select the app")
        st.stop()

    # 選択されたアプリケーションを処理する関数を呼び出す
    render_func = apps[selected_app_name]
    render_func()


def ai_learning():
    learning.ai_learning()

def ai_learning_novelup():
    learning_n.ai_learning_novelup()

def ai_uploadfile():
    upload.upload_file()

def ai_generate():
    generate.ai_generate()

def text_wordcloud():
    if "story" not in st.session_state:
        st.write("テキストがないよ")
        st.stop()

    analyze.create_wordcloud(st.session_state.story)

def text_analyze():
    st.write("工事中")


def text_summarize():
    st.write("工事中")


if __name__ == "__main__":
    main()


