python Classifier/train.py \
  --model_name "microsoft/deberta-v3-large" \
  --data_path "datasets/clusters/final_kmeans/replaced_label_data_kmeans.json" \
  --output_dir classifier/logs/kmeans \
  --batch_size 8 \
  --accumulation_steps 4 \
  --learning_rate 1e-5 \
  --dropout_rate 0.3 \
  --epochs 15 \
  --gamma 2.0
