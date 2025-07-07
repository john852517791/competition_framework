# python predict.py --config a_logs/efficientnet-b4/version_3/pred.yaml \
# --checkpoint_path  a_logs/efficientnet-b4/version_3/checkpoints/best-checkpoint-epoch=26-train_loss=0.0168.ckpt \
# --predict_batch 64 \
# --output_csv_path  a_logs/efficientnet-b4/version_3/submit.csv


# CUDA_VISIBLE_DEVICES=5,6  
python predict.py --config a_logs/efficientnet-b7/version_3/pred.yaml \
--checkpoint_path  a_logs/efficientnet-b7/version_3/checkpoints/best-checkpoint-epoch=10-train_loss=0.0537.ckpt \
--predict_batch 240 \
--output_csv_path  a_logs/efficientnet-b7/version_3/submit.csv
