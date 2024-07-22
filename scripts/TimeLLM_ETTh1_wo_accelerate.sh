model_name=SoftPromptModel
llm_model=GPT2
llm_dim=768
train_epochs=100
learning_rate=0.01
llama_layers=32
num_process=8
batch_size=24
d_model=32
d_ff=128
comment='SoftPromptModel-ETTh1'

python run_main.py \
--task_name long_term_forecast \
--is_training 1 \
--root_path ./dataset/ETT-small/ \
--data_path ETTh1.csv \
--model_id ETTh1_512_96 \
--model $model_name \
--data ETTh1 \
--features M \
--seq_len 512 \
--label_len 48 \
--pred_len 96 \
--factor 3 \
--enc_in 7 \
--dec_in 7 \
--c_out 7 \
--des 'Exp' \
--itr 1 \
--d_model $d_model \
--d_ff $d_ff \
--batch_size $batch_size \
--learning_rate $learning_rate \
--llm_layers $llama_layers \
--llm_model $llm_model \
--llm_dim $llm_dim \
--train_epochs $train_epochs \
--model_comment $comment