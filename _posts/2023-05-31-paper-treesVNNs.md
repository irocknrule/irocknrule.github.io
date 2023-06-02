toc: true
classes: wide

## Paper review: Why do tree-based models still outperform deep learning on typical tabular data?

Here I review the paper "Why do tree-based models still outperform deep learning on typical tabular data?" by Grinsztajn et. al [https://hal.science/hal-03723551v2/document] published at NeurIPS 20222. (Note: Figures in this post are from the paper)


Over the last few years of working at AWS, numerous modeling problems fell under the tabular dataset umbrella, infact most of my work entailed creating regression and classification models using tabular datasets comprising of hundreds of features. Whether it is causal modeling in the economics and financial domains, networking rack growth forecasting within the datacenter space or predicting video rebuffering from video Quality of Experience (QoE) data from the Content Delivery Network (CDN) space; tabular datasets are ubiquitous and used everywhere. We would generally default to using tree based modeling techniques to implement our models and even though I sometimes experimented with some basic deep learning, it never quite worked well enough. Recently, i came across this nice paper where the authors dig deeper into why this the case, i.e. why do tree-based models generally outperform deep learning on tabular datasets. 

## Outline
In this paper, the authors mention that tabular specific Deep Learning (DL) architectures are an active research area but lack an established benchmark since there is no standard datasets to compare against. Various types of tabular datasets are also generally smaller when compared to image based and other datasets.

### Contributions
Their primary contributions in this paper are: 

- **New Benchmarks**: The authors create a new benchmark for tabular data and then carry out a comparison. They further investigate why tree based models outperform DL. They compiled 45 tabular datasets from various domains based on a wide variety of criteria and release it to the research community for further studies. 
- **Experiment setup:**
	- Random search on the hyper-parameter space of around 400 iterations per dataset. 
	- Using accuracy for classification and R2 scores for regressions on the selected datasets
	- Models benchmarked: RandomForest, GradientBoostingTrees and XGBoost from sci-kit learn. For DL they selected: a standard Multi-Layer Perceptron (MLP), ResNET i.e MLP with dropout, batch norm and skip connections, FT_Transformer - a simple transformer model with categorical and numerical feature embeddings and SAINT - a transformer model with an embedding module.

## Results
Tree based models are all much better than any of the DL models. The interesting result was that categorical features are not weaknesses in Neural Nets (NNs) on tabular data as performance gap exists even with only numerical features. 

![](/images/Grinsztajn_paper_fig-1.png "Tree based methods regularly outperforming the DL based methods on tabular data for both regression and classifcation tasks.")

## Digging deeper

The authors now ask the question: Why do tree-based models perform better? We observe that the ensemble methods such as bagging (in RF) and boosting (in XGBoost) with weak learners such as decision trees always work better in the random search space. The authors try to find the inductive biases behind this characteristic. 

### Finding 1: NNs are biased to smooth solutions: 
The authors smooth the training set output with a Gaussian kernel smoother of varying lengths which then prevents the models from learning irregular patterns of the target function. The tree based methods show a higher drop in accuracy compared to the NNs indicating that the latter are biased towards the smoother functions. This in turn suggests that the target functions in the datasets are irregular which NNs struggle to fit and is in line with prior research {Rahaman19} indicating that NNs are biased towards low frequency functions. The authors then go on to suggest that adequate regularization and careful optimization may enable NNs to learn irregular patterns, so its not out of the realm of impossible but tree based approaches do this faster and more easily. This is an interesting finding and I especially like the kernel smoothing approach to uncover this feature of NNs.

### Finding 2: Uninformative features affect NNs more
We can calculate feature importances for Random Forests and then drop those features with low scores with minimal affect on accuracy of our tree based model. This is a known feature engineering technique and helps reduce the overall size of the forest or number of decision trees. The authors show that often-times, removing almost the half the features does not affect the overall accuracy of a GBT but NNs with MLP like architectures are not robust to such low importance features. They show that *adding* these uninformative features results in a higher performance gap between MLPs and the other models. Tabular datasets generally have a high percentage of such uninformative features, so the authors argue that these result in worse accuracies for NN based models.

![](/images/Grinsztajn_paper_fig-4.png "Decrease in accuracy while adding uninformative features.")

### Finding 3: Tabular data is non invariant by rotation, so should be the learners.

When we say that data is invariant by rotation, it means that the data remains unchanged or unaffected when rotated around a specific axis or point. This also means that the data exhibits the same characteristics or properties regardless of its orientation or angle of rotation. 

The authors say that based on {NG04} a NN with a MLP is rotationally invariant, meaning that applying a rotation (unitary matrix) to the features on both the training and testing sets does not change the learning procedure. {NG04} establishes a theoretical link between rotationally invariant learning procedures and uninformative features. It shows that such procedures have a worst-case sample complexity that grows linearly with the number of irrelevant features. Thus MLPs are more hindered by uninformative features because of their rotationally invariant nature, which requires additional steps to identify and remove irrelevant features compared to other models.

## My thoughts and learnings:

- **Dataset creation** : They prune the datasets to 10K samples, remove missing data, remove high cardinality categorical features, but these are generally always the case in tabular datasets. By carrying out these data cleaning exercises, while i understand it being somewhat necessary; simplifies the lay-of-the-land significantly. However it is not a huge blocker towards reading into the results as is. 
- NN optimizations: The authors agree that neural networks can be carefully optimized to perform similar to trees but do not delve into why its not an optimal direction. Downsides of higher training time, more resources, over-training and so on could have been mentioned to better motivate the issue here. 
- The reasons behind the inductive biases uncovered by the authors are certainly very interesting. While i was surprised to learn that NNs are biased towards smooth solutions (i always thought a NN could learn anything), it does make sense when we realize how important learning rates are during the training process in a NN. Larger rates makes the loss go haywire and make finding the local minima much harder (and in many cases impossible). 
- The assertion that learners should be non rotationally invariant is big. Questions arise as to how we can design learners which are naturally non rotationally invariant? I'll read the paper referred here {NG04} to understand the consequences here better and post a review.


Overall this paper was surely a good read with some effective contributions. I liked the experiments and benchmarking set up along-with providing sufficient evidence that with tabular data its best to go along with the ensemble models instead of spending (and potentially wasting) time designing a DL based approach; something which I have certainly been guilty of in the past.



## References

{NG04}: Andrew Y. Ng. Feature selection, L 1 vs. L 2 regularization, and rotational invariance. In Twenty-First  International Conference on Machine Learning - ICML ’04, page 78, Banff, Alberta, Canada,  2004. ACM Press. doi: 10.1145/1015330.1015435.

{Rahaman19}: Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred A. Hamprecht, Yoshua Bengio, and Aaron Courville. On the Spectral Bias of Neural Networks, May 2019.
