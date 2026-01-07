python Classifier/train.py \
  --model_name "answerdotai/ModernBERT-large" \
  --data_path "datasets/clusters/final_kmeans/replaced_label_data_kmeans.json" \
  --output_dir classifier/logs/kmeans \
  --batch_size 8 \
  --accumulation_steps 8 \
  --learning_rate 1e-4 \
  --dropout_rate 0.5 \
  --epochs 15 \
  --gamma 2.0