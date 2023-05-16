Kaggle has a large number of competitions of various types where there is generally a ready dataset of input data provided to help build models. While numerous people use the platform to compete, one of the other useful consequences of having such a platform is the opportunity to work on all these different types of problems with little ground-work needed in terms of data gathering. That is why i simply love working on the platform to identify different useful problems, coming up with solutions to learn on the way (and not necessarily play to win).

Here we look at the first step in a really interesting problem called [**Benetech - Making Graphs Accessible**](https://www.kaggle.com/competitions/benetech-making-graphs-accessible/overview). The goal statement of this competition is: *"The goal of this competition is to extract data represented by four types of charts commonly found in STEM textbooks. You will develop an automatic solution trained on a graph dataset. Your work will help make reading graphs accessible to millions of students with a learning difference or disability."*

What was interesting for me when I looked at this competition was that working on this particular competition involved various aspects of ML and Deep Learning which could/needed to be all tied together in to a workflow. For eg; this problem requires image classification, object detection and Object Character Recognition (OCR) all together to come up with a solution. In a later post, I will describe the overall workflow towards solving this problem but as a start I wanted to describe the simplest first step of classifying the input images by coming up with a simple classifier using the DL libraries of [fastai](https://course.fast.ai/).

This is yet-another-post in the thousands of online posts about image classifiers but there was a small experiment I always wanted to run and then talk about regarding using the pre-trained model weights which are of popular use in the image learning community. I was mainly interested to note the differences between the various *resnet* models and try to come up with a rule of thumb of when to use which. Granted we can always use the latest and greatest iteration of a model, but do we actually need to use the largest pre-trained model in all scenarios and spend a huge amount of time training (goes without saying that larger models need more time to train and require greater CPU/GPU resources)?

Lets look at this simple problem and run some experiments. 

This [notebook](https://github.com/irocknrule/kaggle/blob/main/Bentech-Graphs/classify_images_resnet18_10epochs.ipynb) contains all the code referenced below.

## Input images

As specified in the competition description, there are 4 types of graphs present in the training dataset:
- Bar graphs - vertical
- Bar graphs - horizontal
- Scatter Plots
- Line graphs

As a result, this is indicative for the need of a simple image classifier with output labels being the types listed above. 

### Datablock API
*fastai* provides a very handy library based on the PyTorch ```DataSet``` library to read data which can be used to train our models. Using the input images and corresponding annotations provided in a JSON file, we create a simple dataframe comprising of the image types and the actual paths from where we can read every image.

The DataBlock can now be specified as:

```
graphs = DataBlock(
blocks=(ImageBlock, CategoryBlock),
get_x=ColReader(2),
get_y=lambda o:o.chart,
splitter=RandomSplitter(valid_pct=0.2, seed=42),
item_tfms=Resize(224),
batch_tfms=aug_transforms())

```
Here we create a simple FastAI image datablock based on the dataframe of input images (and their paths + classifications) above, resize the images to 224px, split the dataset randomly to a training/test split of 80/20 and carry out a basic set of image transformations provided by the FastAI library

We can easily view an input batch using the following snippet. Note that the images have been rotated, padded and basically *transformed* using fastai's default transformations.

```
dls = graphs.dataloaders(working_df)
dls.show_batch()
```
![](/images/show_batch_input.png "Sample batch of input images with transforms")

## Training the model
Training the image classifier is extremely easy in *fastai*. We create a new ```vision_learner``` object, provide the input dataloader and the pre-trained model weights along-with the metric to train  by. Here we specify that we want to train for only 10 epochs. 

```
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(10)
```

*fastai's* `fine_tune()` function abstracts away a bunch of steps and makes things so much easier. It takes the pre-trained model weights (in this case *resnet18*), unfreezes the last few layers of the neural net, finds the optimal learning rate and trains for the number of epochs we ask it to train the new model. By transferring all the previous learnings, ie. the weights from the model training we do not need to train the entire model again and save on valuable time/processing resources. 

After training for 10 epochs, we see that the error rate is already quite low, i.e. < 0.1%.

The confusion matrix for the training dataset is below:

![](/images/resnet18_confusion-matrix.png "Confusion matrix from training the model for 10 epochs using resnet18")

## Comparisons with resnet34

We carry out a quick (and dirty) comparison with the more complex *resnet34* image model. I simply train both models for 4 epochs and check the time per epoch and error rates. The average results are listed below:

| Model    | Average Time Per Epoch (in mins) | Error Rate | # of epochs |
|----------|----------------------------------|------------|-------------|
| resnet18 | 3:03                             | 0.001981   | 4           |
| resnet34 | 4:58                             | 0.001568   | 4           |
| resnet18 | 3:09                             | 0.001651   | 10          |

We see that *resnet18* trains faster since its a less complext model and provides a similar accuracy in 10 epochs of training than that from *resnet34* in 4 epochs, which in-turn takes about 2 minutes more to train per epoch.


## Final Thoughts

I am not making any sort of recommendation here, just running a quick test between the models. The larger models will very obviously run better when we are trying to classify larger and more complex images, but for our use case of classifying graph images a simple resnet18 model should suffice. 

The link to the notebook with the resnet34 model is [here](https://github.com/irocknrule/kaggle/blob/main/Bentech-Graphs/classify_images_resnet34.ipynb)

Overall, this first step towards image classification shows us how easy it is to get an image classifier up and running using *fastai*. In upcoming posts, we will discuss the object detection and OCR prediction models we created as part of this Kaggle competition.
