# -*- coding: utf-8 -*-

import streamlit as st
import ai_learning as learning
import ai_generate as generate

def main():

    # アプリケーション名と対応する関数のマッピング
    apps = {
        '-': None,
        'AIテキスト学習': ai_learning,
        'AIテキスト生成': ai_generate,
    }
    selected_app_name = st.sidebar.selectbox(label='apps',
                                             options=list(apps.keys()))

    if selected_app_name == '-':
        st.info('Please select the app')
        st.stop()

    # 選択されたアプリケーションを処理する関数を呼び出す
    render_func = apps[selected_app_name]
    render_func()

def ai_learning():
  learning.ai_learning()

def ai_generate():
  generate.ai_generate()

if __name__ == '__main__':
    main()