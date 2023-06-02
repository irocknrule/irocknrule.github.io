## Paper review: Why do tree-based models still outperform deep learning on typical tabular data?

Here I review the paper "Why do tree-based models still outperform deep learning on typical tabular data?" by Grinsztajn et. al [https://hal.science/hal-03723551v2/document] published at NeurIPS 20222. (Note: Figures in this post are from the paper)


Over the last few years of working at AWS, numerous modeling problems fell under the tabular dataset umbrella, infact most of my work entailed creating regression and classification models using tabular datasets comprising of hundreds of features. Whether it is causal modeling in the economics and financial domains, networking rack growth forecasting within the datacenter space or predicting video rebuffering from video Quality of Experience (QoE) data from the Content Delivery Network (CDN) space; tabular datasets are ubiquitous and used everywhere. We would generally default to using tree based modeling techniques to implement our models and even though I sometimes experimented with some basic deep learning, it never quite worked well enough. Recently, i came across this nice paper where the authors dig deeper into why this the case, i.e. why do tree-based models generally outperform deep learning on tabular datasets. 

## Outline
In this paper, the authors mention that tabular specific DL architectures are an active research area but lack an established benchmark since there is no standard datasets to compare against. Various types of tabular datasets are also generally smaller when compared to image based and other datasets.

### Contributions
Their primary contributions in this paper are: 

- **New Benchmarks**: The authors create a new benchmark for tabular data and then carry out a comparison. They further investigate why tree based models outperform DL. They compiled 45 tabular datasets from various domains based on a wide variety of criteria and release it to the research community for further studies. 
- **Experiment setup:**
	- Random search on the hyper-parameter space of around 400 iterations per dataset. 
	- Using accuracy for classification and R2 scores for regressions on the selected datasets
	- Models benchmarked: RandomForest, GradientBoostingTrees and XGBoost from sci-kit learn. For DL they selected: MLP, Resnet i.e MLP with dropout, batch norm and skip connections, FT_Transformer - a simple transformer model with categorical and numerical feature embeddings and SAINT - a transformer model with an embedding module.

## Results
Tree based models are all much better than any of the DL models. Interesting result was that categorical features are not weaknesses in NNs on tabular data as performance gap exists even with only numerical features. 

(insert figure 1)

## Digging deeper

The authors now ask the question: Why do tree-based models perform better? We observe that the ensemble methods such as bagging (in RF) and boosting (in XGBoost) with weak learners such as decision trees always work better in the random search space. The authors try to find the inductive biases behind this characteristic. 

### Finding 1: NNs are biased to smooth solutions: 
The authors smooth the training set output with a Gaussian kernel smoother of varying lengths which then prevents the models from learning irregular patterns of the target function. The tree based methods show a higher drop in accuracy compared to the NNs indicating that the latter are biased towards the smoother functions. This in turn suggests that the target functions in the datasets are irregular which NNs struggle to fit and is in line with prior research (rahaman 2019 ) indicating that NNs are biased towards low frequency functions. The authors then go on to suggest that adequate regularization and careful optimization may enable NNs to learn irregular patterns, so its not out of the realm of impossible but tree based approaches do this faster and more easily. 

## My thoughts:

- **Dataset creation** : They prune the datasets to 10K samples, remove missing data, remove high cardinality categorical features, but these are generally always the case in tabular datasets. By carrying out these data cleaning exercises, while i understand it being somewhat necessary; simplifies the lay-of-the-land significantly. However it is not a huge blocker towards reading into the results as is. 
- NN optimizations: The authors agree that neural networks can be carefully optimized to perform similar to trees but do not delve into why its not an optimal direction. Downsides of higher training time, more resources, over-training and so on could have been mentioned to better motivate the issue here. 
