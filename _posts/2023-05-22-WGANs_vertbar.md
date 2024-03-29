# Generating images using Generative Adversarial Networks (GANs) 

## Introduction
Generative models are currently extremely popular with text based models such as ChatGPT, Google Bard and so on capturing everyone's attention along-with image models such as DALL-E and Stable Diffusion using text prompts to generate images on the fly. While trying to learn more about how these models work, I came across a wonderful intro tutorial to Generative Adversarial Networks (GANs) in the MIT Deep Learning intro video series (1). Going down the GAN rabbit-hole as a side project while working on the **Benetech - Making Graphs Accessible** (https://www.kaggle.com/competitions/benetech-making-graphs-accessible/overview) competition was a whole lot of fun and while I only scratched the surface, it has been extremely enjoyable to dive deep and learn more of this exciting domain. 

In this post I will detail a very preliminary (and rudimentary) implementation of an image GAN using fastai using the readily available Benetech dataset (since it was already there) and set the scene for more experiments down the line. 

## Generative Adversarial Networks (GANs)
There's a wealth of literature already on GANs, so there's no need to go deep into the descriptions. At a high level, Ian Goodfellow developed this class of models where we essentially train two models at the same time - a generator model which generates data and a discriminator model (also called a critic) which decides if the data the generator decides is fake or not. The discriminator's output is propagated back to the generator which uses that feedback to keep training to improve its model and generate more realistic data. The goal of the generator is to come up with data realistic enough that the discriminator is 'fooled' into believing that it is part of the original dataset. 

Lets consider the discriminator and generator design separately and consider an image GAN, i.e. a GAN to generate images from an input set of images.

### Generator
The job of the generator model is to simply generate images which are closest (with the lowest loss) to the set of training images. However the set of training images is not provided to the generator, so how does it start generating images? It simply starts by generating random noise as the input to the generator network and outputs these images. This output of a random generated image is then passed to the discriminator model which classifies it as fake/real and generates the *generator loss* value. By using standard Backpropagation, we update the generator weights based on a learning rate and training continues with the goal of reducing generator loss. 

### Discriminator
The discriminator is trained to provide the generator the feedback that the generated image is fake or real. However the discriminator itself needs to be trained to be able to provide good enough feedback to the generator, so we train the discriminator from the set of real images coming from the training set. The discriminator is a trained classifier where it decides (based on its input training set of real images) if the generator's image is real or fake and what is the loss value for the generated image. Since the discriminator is a simpler image classifier, it does not need very extensive training and we can use transfer learning easily by re-using any standard pre-trained model such as *imagenet* or *resnet*. 

### Adversaries
The generator and discriminator models are now *adversaries* as the generator's goal is to fool the discriminator by generating real-enough fake data. The discriminator's goal is to not be fooled and keep identifying the fake images so indirectly induce the generator to get better and better. Both models keep training, the discriminator from the real set of images and the generator from the feedback from the discriminator and its own set of fake images. 

## Generating a basic GAN
With the description of a GAN out of the way, lets look at creating a very basic GAN using *fastai*. As mentioned earlier, I have been working with a large set of graph images and the ready dataset looked to be an ideal candidate to throw into fastai to see what it can generate. Note that as in any modeling exercise, there are a lot of gotchas and to-do's to get *good* results, but here as a starting point I am not looking to come up with an optimal GAN with the best results. 

### *fastai* GAN module
*fastai* contains the GAN module within its vision library which can be accessed with:

`from fastai.vision.gan import *`

I wanted to quickly point out the the GAN learner source code:

```
Source:        
class GANLearner(Learner):
    "A `Learner` suitable for GANs."
    def __init__(self,
        dls:DataLoaders, # DataLoaders object for GAN data
        generator:nn.Module, # Generator model
        critic:nn.Module, # Critic model
        gen_loss_func:callable, # Generator loss function
        crit_loss_func:callable, # Critic loss function
        switcher:Callback|None=None, # Callback for switching between generator and critic training, defaults to `FixedGANSwitcher`
        gen_first:bool=False, # Whether we start with generator training
        switch_eval:bool=True, # Whether the model should be set to eval mode when calculating loss
        show_img:bool=True, # Whether to show example generated images during training
        clip:None|float=None, # How much to clip the weights
        cbs:Callback|None|MutableSequence=None, # Additional callbacks
        metrics:None|MutableSequence|callable=None, # Metrics
        **kwargs
    ):
       ...
       ...
       
    @classmethod
    def wgan(cls,
        dls:DataLoaders, # DataLoaders object for GAN data
        generator:nn.Module, # Generator model
        critic:nn.Module, # Critic model
        switcher:Callback|None=None, # Callback for switching between generator and critic training, defaults to `FixedGANSwitcher(n_crit=5, n_gen=1)`
        clip:None|float=0.01, # How much to clip the weights
        switch_eval:bool=False, # Whether the model should be set to eval mode when calculating loss
        **kwargs
    ):
        "Create a [WGAN](https://arxiv.org/abs/1701.07875) from `dls`, `generator` and `critic`."
        if switcher is None: switcher = FixedGANSwitcher(n_crit=5, n_gen=1)
        return cls(dls, generator, critic, _tk_mean, _tk_diff, switcher=switcher, clip=clip, switch_eval=switch_eval, **kwargs)
```

The `wgan()` function presents the implementation of the WGAN network and the associated paper regarding Wasserstein GAN is provided at \[2\]. It is an extremely interesting paper and I plan on posting a review of the paper soon (once i post the review, I will update the link here).

### Experiment setup
In the Kaggle Benetech competition, the training set contains almost ~7k images of vertical bar graphs, so we use that dataset in our experiment. 

We set a batch size of 128 and re-size our images to 64x64 so that we can train faster (GANs take a long time to train, we discuss it later on in this post). We now create a basic fastai datablock where the inputs are simply noise for the generator and the training dataset for the discriminator is loaded into the datablock from the given path by the standard `get_image_files` *fastai* helper function.

```
dblock = DataBlock(blocks = (TransformBlock, ImageBlock),
get_x = generate_noise,
get_items = get_image_files,
splitter = IndexSplitter([]),
item_tfms=Resize(size, method=ResizeMethod.Crop),
batch_tfms = Normalize.from_stats(torch.tensor([0.5,0.5,0.5]), torch.tensor([0.5,0.5,0.5])))
```

Viewing an instance of the dataloader gives us:

`dls.show_batch(max_n=16)`

![](/images/show_batch_input.png "An input batch of graph pictures being used for training. These images are from the Benetech Kaggle competition.")

### Model training
We now create the generator and discriminator models (*fastai* uses critic, so we use that here):

```
generator = basic_generator(64, n_channels=3, n_extra_layers=1)
critic = basic_critic (64, n_channels=3, n_extra_layers=1, 
                       act_cls=partial(nn.LeakyReLU, negative_slope=0.2))

```

An interesting note here is that the *LeakyReLU* activation function is used by the critic as the last layer. This is to guard against the *dying ReLU* problem where the neurons can get perpetually stuck outputting 0s for all inputs in a regular ReLU. Leaky ReLUs allow a small negative gradient to pass through thereby letting some information through the network to the generator. Some more very useful information is provided at \[3\].

We now create the model and train for a few epochs to observe the start of the image generation process. 

```
learn = GANLearner.wgan(dls, generator, critic, opt_func = RMSProp)
learn.fit(5, 2e-4, wd=0.)
```

### Model Outputs
After a quick training session of 5 epochs, we check our results:

`learn.show_results(max_n=9, ds_idx=0)`

![](/images/vertbar_5-epochs.png "Training results for vertical bars after only 5 epochs. Note that in 5 epochs the random noise is trending away.")


We can see that there is almost nothing that the generator has learned at this point in time, so the generated images are simply random noise with some initial patterns showing up. Since the overall batch size is 64 and the generated images are small (64x64) along with not having too many features in the dataset, we train for a further 500 epochs. Checking the results now shows some surprisingly cool results.

![](/images/vertbar_250-epochs.png "Training results for vertical bars after 500 epochs. The overall structure of the vertical bar images have started showing up now.")


We can now see that the general structure of the bar charts is coming together. We have text at the top for the chart title, the axes labels and titles are showing up and the vertical bars in different colors. There is no specific values or properly legible text for the labels and titles but the fact that the overall structure of these graphs are being generated after training for only a couple hours is really nice. 

## Conclusions
GANs require a significant amount of training to generate realistic outputs with the timing needs dependent on the type of input pictures, their sizes, complexity and a bunch of other factors. As I dive deeper into this domain, I will write a few more posts with some really interesting results from current experiments which require training on the order of days. This was just an initial foray into this extremely interesting form of generative AI specifically using out of the box *fastai* code, again with little or no modifications. There are a bunch of experimentation currently in progress with different architectures which I will describe in a new post.

The link to the notebook with the code discussed is here: (https://github.com/irocknrule/kaggle/blob/main/Bentech-Graphs/generate_graphs.ipynb)

## References
(1) MIT Intro to Deep learning - Lecture 3: [Deep Generative Modeling](https://www.youtube.com/watch?v=3G5hWM6jqPk)

{2} Wasserstein GAN: https://arxiv.org/abs/1701.07875

{3} A Short Introduction to Generative Adversarial Networks (https://sthalles.github.io/intro-to-gans/)

