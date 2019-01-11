# [kaggle competition protein classification](https://www.kaggle.com/c/human-protein-atlas-image-classification)

I worked on the competition for about two weeks, and got ~100th place in private lb. 

Many thanks to [Spytensor's](https://www.kaggle.com/spytensor) [awsome starter code](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/72812) to start with, and gradually changing and adding my stuff to it.  

My final model is just a single resnet50(on a 5 folds setting), trained on rgb data(with hpa set). Thresholding is done on the out of fold predictions, using a single threshold rather than per label thresholds. TTA is done by averaging outputs of 10 randomly transformed version of the same image. 

Some stuffs that I found useful:

1. Use RGB instead of RGBY: I found at the end of first week that yellow channel looks very similar to red or blue, therefore should be kind of redundent, not to mention that I don't know if the yellow channel from hpa dataset trustful or not.
2. Use HPA dataset: I found adding hpa data into training data splits, can improve validtion results on my validation set from Kaggle data.
3. Use weighted sampler in dataloader: Cause the model to see rare labels a bit more often than not using the sampler. I found this is better than use f1_loss with uniform randomly sampled data. 
4. Use TTA: improve ~ .01 for most models, and for my last fold, it improves a lot. 
