python Classifier/train.py \
  --model_name "microsoft/deberta-v3-large" \
  --data_path "datasets/clusters/final_agglomerative/replaced_label_data_agglomerative.json" \
  --output_dir classifier/logs/agglomerative \
  --batch_size 8 \
  --accumulation_steps 4 \
  --learning_rate 1e-5 \
  --dropout_rate 0.0 \
  --epochs 15 \
  --gamma 2.0


