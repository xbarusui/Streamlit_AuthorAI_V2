# -*- coding: utf-8 -*-

import streamlit as st
import subprocess
from subprocess import PIPE
import tempfile
from pathlib import Path
from transformers import T5Tokenizer, AutoModelForCausalLM
import tempfile

def ai_lerning():

    uploaded_file = st.file_uploader("upload file", type={"txt"})
    temp_dir = ""
    if uploaded_file is not None:
        #directory作成
        temp_dir = tempfile.mkdtemp()
        print('temp dir path:', temp_dir)

        # Make temp file path from uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
            filepath = Path(temp_file.name)
            filepath.write_text(uploaded_file.getvalue().decode('utf-8').replace("\n",""))
            st.write(temp_file.name) #write_bytes
            st.write(filepath.read_text())
            st.success("Saved File:{} to tempDir".format(temp_file.name))

        st.session_state.session_dir = temp_dir
        st.write('session_state.session_dir = ' + str(st.session_state.session_dir))

        st.write("ファインチューニングはいるのはいつ？")
        # ファインチューニングの実行
        run_command(["python3","gpt2japanese_study.py", 
                "--model_name_or_path=rinna/japanese-gpt2-xsmall", 
                "--train_file="+temp_file.name, 
                "--validation_file="+temp_file.name, 
                "--do_train",
                "--do_eval",
                "--num_train_epochs=10", 
                "--save_steps=5000",
                "--save_total_limit=3",
                "--per_device_train_batch_size=1", 
                "--per_device_eval_batch_size=1",
                "--output_dir="+st.session_state.session_dir,
                "--use_fast_tokenizer=False",
                "--overwrite_output_dir"])

        st.write("ファインチューニングおわり")


def ai_generate():
    """トップ！のアプリケーションを処理する関数"""
    
    noveltext = st.text_area(label='Multi-line message', height=275)
    # バリデーション処理
    if len(noveltext) < 1:
        st.warning('Please input your text')
        # 条件を満たないときは処理を停止する
        st.stop()

    if 'session_dir' not in st.session_state: 
        st.session_state.session_dir = "rinna/japanese-gpt2-xsmall" #session_dirがsession_stateに追加されていない場合，元モデルで初期化（不要？）

    # トークナイザーとモデルの準備
    tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-xsmall")
    model = AutoModelForCausalLM.from_pretrained(st.session_state.session_dir)

    novellength=256
    novelseq=3

    # 生成された文章を表示します。
    st.write("冒頭の文章")
    st.write("----------------------------------------")
    st.write(noveltext)
    st.write("----------------------------------------")

    # 推論
    input = tokenizer.encode( noveltext , return_tensors="pt")
    output = model.generate(input, do_sample=True, max_length=novellength, num_return_sequences=novelseq,top_k=0,min_length=int(novellength/2))
    # 出力されたコードを文章に戻します。
    DecodedOutput = tokenizer.batch_decode(output)

    for j in range(novelseq):
      st.write("--- AI生成文章" + str(j+1) + "回目 ---")
      i = 0
      while i < len(DecodedOutput[j]):
        st.write(DecodedOutput[j].replace('</s>','')[i:i+80]) 
        i = i+80
      st.write("----------------------------------------")


def main():
    # アプリケーション名と対応する関数のマッピング
    apps = {
        '-': None,
        'AIテキスト学習': ai_lerning,
        'AI自動生成': ai_generate,
    }
    selected_app_name = st.sidebar.selectbox(label='apps',
                                             options=list(apps.keys()))

    if selected_app_name == '-':
        st.info('Please select the app')
        st.stop()

    # 選択されたアプリケーションを処理する関数を呼び出す
    render_func = apps[selected_app_name]
    render_func()
  

def run_command(args):
    """Run command, transfer stdout/stderr back into Streamlit and manage error"""
    st.info(f"Running '{' '.join(args)}'")
    result = subprocess.run(args, capture_output=True, text=True)
    try:
        result.check_returncode()
        st.info(result.stdout)
    except subprocess.CalledProcessError as e:
        st.error(result.stderr)
        raise e


if __name__ == '__main__':
    main()