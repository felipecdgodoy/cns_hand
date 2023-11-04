alpha=0.005
epochs=10
lamb=0
batch_size=8
wd=0
num_folds=5

python3 train_repr.py --lr $alpha --epochs $epochs --lamb $lamb --batch_size $batch_size --wd $wd --num_folds $num_folds