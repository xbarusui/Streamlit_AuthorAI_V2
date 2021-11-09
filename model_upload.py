# -*- coding: utf-8 -*-

import streamlit as st
import tempfile
import chardet
import shutil
import zipfile

from pathlib import Path
from transformers import T5Tokenizer, AutoModelForCausalLM
from transformers import TextDataset,DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments,AutoModelWithLMHead

def model_upload():

    # トークナイザーとモデルの準備
    tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-xsmall")

    # ヘッダ
    st.header("AI モデルアップロード")

    st.write("AI モデルのアップロードをしてください")

    uploaded_file = st.file_uploader("upload file")

    temp_dir = ""
    if uploaded_file is not None:
        #directory作成
        temp_dir = tempfile.mkdtemp()

        filepath = ""
        # Make temp file path from uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as f:

            filepath = Path(f.name)
            f.write(uploaded_file.getvalue())

        #zip ファイルの解凍とファイルの配置
        with zipfile.ZipFile(filepath) as existing_zip:
            existing_zip.extractall(Path(temp_dir))


        st.success("Saved File:{} to tempDir".format(f.name))
#        st.session_state.story = filepath.read_text()

        st.session_state.session_dir = temp_dir
        st.write('session_state.session_dir = ' + str(st.session_state.session_dir))
        st.write("f.name = " + str(f.name))