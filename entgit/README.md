main.py implements training and validation of chosen number of runs on a chosen setting. It has functions to plot the tracked performance of the model through the dictionary history. Before the call to main ends, the best initialization weights are found and then saved in models/inits/best.pth.  

fold.py implements k-cross validation, using the dataloaders of dataloader.py. Before fold.main() is called, a model should be placed into models/inits/ and named best.pth, which main.main() does with the best weight initialization. fold.main() will save the final weight values of each model in models/folds. 

dataloader.py implements functions that output pytorch dataloaders. To do so, it also implements a class that inherits from pytorch's Dataset class to be used with the pytorch's random samplers. This custom class expects the images to be in Data/All/_SpecificClass_, for example, Data/All/IP.   dataloader.get_loaders() returns a dictionary of dataloaders which correspond to the training, validation, and testing dataloaders. dataloader.get_fold_loaders() returns a list whose elements are dictionaries that contain either training or validation dataloaders. This function is used in fold.py.  

model.py implements functions that return pretrained resnet152 models given some specifications, and the corresponding optimizer that considers the frozen layers of a model. For instance, calling mod = model.get_pretrained_model([4]) returns a model whose only layer not frozen is the 4th layer. Then model.get_optimizer(mod) can be called which returns an optimizer that only keeps track of the 4th layer weights. 

traintest.py implement functions related to training and testing a model. traintest.trainepoch() trains the given model for an epoch using the given criterion, optimizer and dataloader, and then validates its performance using the validation dataloader. traintest.test() merely tests the given model with the data from the given dataloader. traintest.fulltrain() calls traintest.trainepoch() the given number of times and saves the weights of the final version.  

roc.py uses the saved dataloaders and models from fold.main() to calculate the tprs (true positive rates) and other metrics. The tprs are saved so that they can be used in plot_roc.py. plot_roc uses the saved tprs to calculate the aucs, the standard deviations of aucs, the mean tprs, and the standard deviation of tprs, and then uses these calculations to plot them. The performances of the fellows and residents are also plotted along side the appropiate plots. Thus roc.main() should be run before plot_roc.py is run.  

stopping_criteriorn.py implements a class that keeps track of the metrics necessary for the stopping criterion. It is used in the main.py file.    

sweep.yaml is an example of a yaml file that can be used with wandb for an easy implementation of grid search.  

Example: 
Call main.main(). Use the best_ind from this in a call to fold.main(). This should generate and save the number of specified models and loaders, so that roc.main() can be called. Then plot_roc.py can be called and the appropiate graphs should be saved. 
