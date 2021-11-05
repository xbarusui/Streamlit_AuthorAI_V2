# -*- coding: utf-8 -*-

import streamlit as st
from transformers import T5Tokenizer, AutoModelForCausalLM

def ai_generate():

    # トークナイザーとモデルの準備
    tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-xsmall")

    # タイトル
    st.title('創作作家AI')
    # ヘッダ
    st.header("AI テキスト生成")

    st.write("出力文字数と何回生成するかを指定して、テキスト先頭の文字数を入力する")

    st.write("最初の文章が長い方が出力される文章は長くなる傾向にあります。")
    
    novellength = st.number_input(label='生成する最大文字数',value=256,)

    novelseq = st.number_input(label='生成回数',value=3,)


    noveltext = st.text_area(label='Multi-line message', height=275)
    # バリデーション処理
    if len(noveltext) < 1:
        st.warning('Please input your text')
        # 条件を満たないときは処理を停止する
        st.stop()

    if 'session_dir' not in st.session_state: 
        st.session_state.session_dir = "rinna/japanese-gpt2-xsmall" #session_dirがsession_stateに追加されていない場合，元モデルで初期化（不要？）

    # トークナイザーとモデルの準備
    model = AutoModelForCausalLM.from_pretrained(st.session_state.session_dir)


    # 生成された文章を表示します。
    st.write("冒頭の文章")
    st.write("----------------------------------------")
    st.write(noveltext)
    st.write("----------------------------------------")

    # 推論
    input = tokenizer.encode( noveltext , return_tensors="pt")
    output = model.generate(input, do_sample=True, max_length=novellength, num_return_sequences=novelseq,top_k=0,min_length=int(novellength/2),no_repeat_ngram_size=3)
    # 出力されたコードを文章に戻します。
    DecodedOutput = tokenizer.batch_decode(output)

    for j in range(novelseq):
      st.write("--- AI生成文章" + str(j+1) + "回目 ---")
      i = 0
      while i < len(DecodedOutput[j]):
        st.write(DecodedOutput[j].replace('</s>','')[i:i+novellength]) 
        i = i+novellength
      st.write("----------------------------------------")
