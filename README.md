# main

main.py implements training and validation of chosen number of runs on a chosen setting. 
fold.py implements k-cross validation, using the dataloaders of dataloader.py.
sweep.yaml is an example of a yaml file that can be used with wandb for an easy implementation of grid search.
stopping_criteriorn.py implements a class that keeps track of the metrics necessary for the stopping criterion.
model.py supplies functions to grab a pre-trained model with the given specifications.
roc.py should be run before plot_roc.py, so that roc.py can generate the necessary tprs for plot_roc.py. 

Example: 
Call main.main(). Use the best_ind from this in a call to fold.main(). This should generate and save the number of specified models and loaders, 
so that roc.main() can be called.


