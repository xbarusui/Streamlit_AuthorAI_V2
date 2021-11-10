# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import novel_downloader
from transformers import T5Tokenizer, AutoModelForCausalLM
from transformers import TextDataset,DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments,AutoModelWithLMHead


def ai_learning_novelup():

    # ヘッダ
    st.header("AI テキスト学習（ノベプラから取得）")

    st.write("epoch数を指定して学習させたい文章のテキストをアップロードしてください")

    st.write("エポックというのは訓練データを学習する単位になります。初期値は 10 ですが、1-10程度に設定してください。学習が少ないですが30分越えるとリロードして利用できなくなります")

    epochnum = st.number_input(label="Epoch",value=10)

    st.write("ノベプラのURLを入れてね　例：https://novelup.plus/story/942595339")

    name = st.text_input(label="URL",key="textbox")
    # バリデーション処理
    if len(name) < 1:
        st.warning('URLを指定してください')
        # 条件を満たないときは処理を停止する
        st.stop()

    namelist = []
    load_data = []
    #directory作成
    #st.session_state.model_dir = tempfile.mkdtemp()
    st.session_state.model_dir = tempfile.mkdtemp("","",st.session_state.content_dir)

    st.write("session_state.model_dir = " + str(st.session_state.model_dir))

    #元々複数でリスト渡しだったので少し修正
    namelist =[name]

    df = pd.DataFrame(novel_downloader.nobera_download(namelist))
    df["text"] = df["text"].str.replace("\n","")
    df["text"] = df["text"].str.replace("　","")
    load_data = ''.join(df["text"])

    # Make temp file path from uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:

        filepath = Path(f.name)
#        filepath.write_text(uploaded_file.getvalue().decode(chardet.detect(uploaded_file.getvalue())["encoding"]).replace("\n",""))
        filepath.write_text(load_data.replace("\n",""))

        with st.expander("アップロードしたテキストを確認したい場合はこちらを開いて下さい"):
            st.write(filepath.read_text())

            st.session_state.story = filepath.read_text()

        st.success("Saved File:{} to tempDir".format(f.name))


    # トークナイザーとモデルの準備
    tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-xsmall")

    status_area = st.empty()
    status_area.info("学習開始")

    train_dataset,test_dataset,data_collator = load_dataset(f.name,f.name,tokenizer)

    model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-xsmall")

    training_args = TrainingArguments(
        do_train=True,
        do_eval=True,
        save_total_limit=3,
        output_dir=st.session_state.model_dir, #The output directory
        overwrite_output_dir=True, #overwrite the content of the output directory
        num_train_epochs=epochnum, # number of training epochs
        per_device_train_batch_size=1, # batch size for training
        per_device_eval_batch_size=1,  # batch size for evaluation
        save_steps=5000, # after # steps model is saved 
        prediction_loss_only=True
        )


    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()

    trainer.save_model()

    status_area.info("学習終了")

    shutil.make_archive(str(st.session_state.model_dir) , "zip", root_dir=str(st.session_state.model_dir))
                
    with open(str(st.session_state.model_dir)+".zip", "rb") as my_file:
        st.download_button(label = 'Download', data = my_file, file_name = str(st.session_state.model_dir)+".zip", mime = "application/octet-stream") 


@st.cache(allow_output_mutation=True, max_entries=10, ttl=3600,suppress_st_warning=True)
def load_dataset(train_path,test_path,tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=128)
     
    test_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=test_path,
          block_size=128)   
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset,test_dataset,data_collator
