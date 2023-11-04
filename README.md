This repo already contains the necessary ```pickle``` files to run experiments, which handled all the pre-processing and patient matching. However, in case you need to run it again you can use ```python3 data_matching_builder.py```

A baseline experiment can be deployed through: ```sh run_baseline.sh```

This command launches an experiment with no disentanglement component, meaning a purely BCE loss for the 4-way classification problem

The parameters inside the bash script can be changed to deploy different experiments. The parameters are:
* alpha (learning rate for the model)
* epochs
* lamb (strength of the disentanglement component)
* batch_size
* wd (weight decay for optimizer)
* num_folds (number of data splits for k-fold cross validation)

An experiment can be launched directly from the command line as well:
```python3 train_repr.py --alpha 0.05 --lamb 0.5 --num_folds 10```

Using the ```train_repr.py``` script will save the following data to the same directory *for each of the folds*:
1. copy of the state dict for the post-trained model
2. list of predictions (two independent probability assignments for each testing image, i.e. [0.87, 0.44])
3. list of recorded testing accuracies during training
4. list of recorded loss values during training
5. list of ids for the patients in the fold
