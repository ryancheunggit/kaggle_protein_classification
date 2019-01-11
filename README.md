# [kaggle competition protein classification](https://www.kaggle.com/c/human-protein-atlas-image-classification)

I worked on the competition for about two weeks, and got ~100th(5%) place in fianl leaderboard. 

Many thanks to [Spytensor's](https://www.kaggle.com/spytensor) [awsome starter code](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/72812), which I used to startwith and gradually changing and adding my stuff to it.  

My final model is just a single resnet50(on a 5 folds setting), trained on rgb data(with hpa set). Thresholding is done on the out of fold predictions, using a single threshold rather than per label thresholds. TTA is done by averaging outputs of 10 randomly transformed version of the same image. 

Some stuffs that I found useful:

1. Use RGB instead of RGBY: I found at the end of first week that yellow channel looks very similar to red or blue, therefore should be kind of redundent, not to mention that I don't know if the yellow channel from hpa dataset trustful or not.
2. Use HPA dataset: I found adding hpa data into training data splits, can improve validtion results on my validation set from Kaggle data.
3. Use mutlilabel stratified splits: Thanks [Trent](https://github.com/trent-b) for his package [iterstrat](https://github.com/trent-b/iterative-stratification) . 
4. Use [weighted sampler](https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html) in PyTorch dataloader: With BCE loss, making the model to see rare labels a bit more often, I found this is better than use f1_loss with uniform randomly sampled data. 
5. Use TTA: It improves ~ .01 for most folds, and for my last fold, it improves a .05(very strange fold, which I have no time to dig into). 

