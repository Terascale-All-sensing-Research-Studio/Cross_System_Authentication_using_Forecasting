seq_len=25
pred_len=40
model_name=PatchTST

model_id_name=general_forecast
data_name=general_forecast

random_seed=2021
for label_len in 5 10 15 20
do
    python3 -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$label_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features all \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --gpu 0 \
      --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$label_len'_'$pred_len.log 
done