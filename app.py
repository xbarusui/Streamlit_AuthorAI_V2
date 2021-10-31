# -*- coding: utf-8 -*-

import streamlit as st
import tempfile
from pathlib import Path
from transformers import T5Tokenizer, AutoModelForCausalLM
from transformers import TextDataset,DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments,AutoModelWithLMHead


def ai_lerning():

    # トークナイザーとモデルの準備
    tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-xsmall")

    # タイトル
    st.title('AI テキスト学習')
    # ヘッダ
    st.header('epoch数を指定して学習させたい文章のテキストをアップロードしてください')

    st.write('エポックというのは訓練データを学習する単位になります。初期値は 30 ですが、それ以上学習させると精度が上がります。が、10-30くらいにしないと遅いです')

    epochnum = st.number_input(label='Epoch',value=30,)

    uploaded_file = st.file_uploader("upload file", type={"txt"})
    temp_dir = ""
    if uploaded_file is not None:
        #directory作成
        temp_dir = tempfile.mkdtemp()

        # Make temp file path from uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
            filepath = Path(temp_file.name)
            filepath.write_text(uploaded_file.getvalue().decode('utf-8').replace("\n",""))
#            st.write(temp_file.name) #write_bytes
            with st.expander("アップロードしたテキストを確認したい場合はこちらを開いて下さい"):
                st.write(filepath.read_text())

            st.success("Saved File:{} to tempDir".format(temp_file.name))

        st.session_state.session_dir = temp_dir
        st.write('session_state.session_dir = ' + str(st.session_state.session_dir))

        status_area = st.empty()
        status_area.info("学習開始")

        # 
        train_dataset,test_dataset,data_collator = load_dataset(temp_file.name,temp_file.name,tokenizer)

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


def ai_generate():

    # トークナイザーとモデルの準備
    tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-xsmall")

    # タイトル
    st.title('AI テキスト生成')
    # ヘッダ
    st.header('出力文字数と何回生成するかを指定して、テキスト先頭の文字数を入力する')

    st.write('最初の文章が長い方が出力される文章は長くなる傾向にあります。')
    
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


if __name__ == '__main__':
    main()