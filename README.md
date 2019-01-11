# [kaggle competition protein classification](https://www.kaggle.com/c/human-protein-atlas-image-classification)

I worked on the competition for about two weeks, rushed to get to ~100th(5%) place in fianl leaderboard. I have quite limited amount of time to work with, both in the sense of remaining competition time and in the sense that I could only code after my daughter finally goes to bed every night. So, instead of running large experiments to see what works, I need to identify what is the next most promising thing to do, by reading posts, looking at data with bare eyes, and running smaller experiments that I can do quickly. Instead of build an army of models, I can only afford to build one model. All these constraints made this competition quite fun for me. 

Many thanks to [Spytensor's](https://www.kaggle.com/spytensor) [awsome starter code](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/72812), which I used to startwith and gradually changing and adding my stuff to it.  

My final model is just a single resnet50(on a 5 folds setting), trained on rgb data(with hpa set). Thresholding is done on the out of fold predictions, using a single threshold rather than per label thresholds. TTA is done by averaging outputs of 10 randomly transformed version of the same image. 

Some stuffs that I found useful:  

1. Use RGB instead of RGBY: By looking at the images I found that yellow channel looks very similar to red or blue, therefore should be kind of redundent to use RGBY, not to mention that I don't even know if the yellow channels from hpa dataset are trustworthy or not.  
2. Use HPA dataset: I found adding hpa data into my Kaggle training data split, can improve validtion results on my Kaggle validation set.  
3. Use mutlilabel stratified splits: Thanks [Trent](https://github.com/trent-b) for his package [iterstrat](https://github.com/trent-b/iterative-stratification). It made my models cross folds performance variation low(except for fold 4 be slightly off). Before this, my cross folds variation was bigger.  
4. Use data augmentation: I think rotate, flip, shear, and random cropping all make sense and used them all with awsome [albumentations](https://github.com/albu/albumentations) package.    
5. Use [weighted sampler](https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html) in PyTorch dataloader: With BCE loss, making the model to see rare labels a bit more often, I found, is better than use f1_loss with uniform randomly sampled data.   
6. Use TTA: It improves ~ .01 for most folds, and for my last fold, it improves about .04(very strange fold, which I have no time to dig into).   

Sample command I used for training the resnet50 model:
```
python main.py --model_name=resnet50 --fold=0 --batch_size=32 --train_batch_per_epoch=800 --valid_batch_per_epoch=448 --split_method=mskf --train_with_hpa=True --valid_with_hpa=True --weighted_samples=True --tta=10
```

