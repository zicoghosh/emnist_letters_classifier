export PYTHONDONTWRITEBYTECODE=1

#python3 -u main.py --classes 26 --epochs 100 --eval_interval 5 --lr 0.001 | tee logs/03_EMNIST_31_05_2021_1417.txt

# only-evaluation
python3 -u main.py --classes 26 --epochs 100 --eval_interval 5 --lr 0.001 --only-evaluation --checkpoint saved_model/EMNIST3/ENGLISH_HW_UC_classfier_model_better_starting_point_learning_rate_0.001_epochs_100_datetime_30_05_2021_04_57_44/final_model.pt