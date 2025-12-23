model_name=PerimidFormer



  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/air/air/ \
  --data_path airquality.csv \
  --model_id air_96_24 \
  --model $model_name \
  --data custom \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --layers 1 \
  --chan_in 12 \
  --d_model 512 \
  --top_k 2 \
  --dropout 0.1 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 2 \
  --learning_rate 0.001 \
  --train_epochs 15 \
  --patience 3

  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/air/air/ \
  --data_path airquality.csv \
  --model_id air_96_48 \
  --model $model_name \
  --data custom \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --layers 1 \
  --chan_in 12 \
  --d_model 512 \
  --top_k 2 \
  --dropout 0.1 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 2 \
  --learning_rate 0.001 \
  --train_epochs 15 \
  --patience 3 


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/air/air/ \
  --data_path airquality.csv \
  --model_id air_96_96 \
  --model $model_name \
  --data custom \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --layers 1 \
  --chan_in 12 \
  --d_model 512 \
  --top_k 2 \
  --dropout 0.3 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 2 \
  --learning_rate 0.001 \
  --train_epochs 15 \
  --patience 3 

  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/air/air/ \
  --data_path airquality.csv \
  --model_id air_96_168 \
  --model $model_name \
  --data custom \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 168 \
  --layers 1 \
  --chan_in 12 \
  --d_model 512 \
  --top_k 2 \
  --dropout 0.3 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 2 \
  --learning_rate 0.001 \
  --train_epochs 15 \
  --patience 3 

