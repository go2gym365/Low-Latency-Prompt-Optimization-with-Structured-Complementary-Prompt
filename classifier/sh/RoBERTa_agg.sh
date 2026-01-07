python Classifier/train.py \
  --model_name "FacebookAI/roberta-large" \
  --data_path "datasets/clusters/final_agglomerative/replaced_label_data_agglomerative.json" \
  --output_dir classifier/logs/agglomerative \
  --batch_size 8 \
  --accumulation_steps 8 \
  --learning_rate 1e-4 \
  --dropout_rate 0.3 \
  --epochs 15 \
  --gamma 2.0

