# -*- coding: utf-8 -*-

import streamlit as st
import tempfile
import chardet
from pathlib import Path
from transformers import T5Tokenizer, AutoModelForCausalLM
from transformers import TextDataset,DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments,AutoModelWithLMHead


def ai_learning():

    # トークナイザーとモデルの準備
    tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-xsmall")

    # ヘッダ
    st.header("AI テキスト学習")

    st.write("epoch数を指定して学習させたい文章のテキストをアップロードしてください")

    st.write("エポックというのは訓練データを学習する単位になります。初期値は 10 ですが、1-10程度に設定してください。学習が少ないですが30分越えるとリロードして利用できなくなります")

    epochnum = st.number_input(label='Epoch',value=10)

    uploaded_file = st.file_uploader("upload file", type={"txt"})

    temp_dir = ""
    if uploaded_file is not None:
        #directory作成
        temp_dir = tempfile.mkdtemp()

        # Make temp file path from uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:

            filepath = Path(f.name)
            filepath.write_text(uploaded_file.getvalue().decode(chardet.detect(uploaded_file.getvalue())["encoding"]).replace("\n",""))

            with st.expander("アップロードしたテキストを確認したい場合はこちらを開いて下さい"):
                st.write(filepath.read_text())

            st.success("Saved File:{} to tempDir".format(f.name))
            st.session_state.story = filepath.read_text()

        st.session_state.session_dir = temp_dir
        st.write('session_state.session_dir = ' + str(st.session_state.session_dir))
        st.write("f.name = " + str(f.name))
        status_area = st.empty()
        status_area.info("学習開始")


        train_dataset,test_dataset,data_collator = load_dataset(f.name,f.name,tokenizer)

        model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-xsmall")

        training_args = TrainingArguments(
            do_train=True,
            do_eval=True,
            save_total_limit=3,
            output_dir=st.session_state.session_dir, #The output directory
            overwrite_output_dir=True, #overwrite the content of the output directory
            num_train_epochs=epochnum, # number of training epochs
#            per_device_train_batch_size=32, # batch size for training
#            per_device_eval_batch_size=64,  # batch size for evaluation
#            per_gpu_train_batch_size=64,
            per_device_train_batch_size=1, # batch size for training
            per_device_eval_batch_size=1,  # batch size for evaluation
#            eval_steps = 400, # Number of update steps between two evaluations.
            save_steps=5000, # after # steps model is saved 
#            warmup_steps=500,# number of warmup steps for learning rate scheduler
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
