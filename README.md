# Equivariance Allows Handling Multiple Nuisance Variables When Analyzing Pooled Neuroimaging Datasets

## Abstract
Pooling multiple neuroimaging datasets across institutions often enables significant improvements in statistical power when evaluating associations (e.g., between risk factors and disease outcomes) that would otherwise be too weak to detect. When there is only a {\em single} source of variability (e.g., different scanners), domain adaptation and matching the distributions of representations may suffice in many scenarios. But in the presence of {\em more than one} nuisance variable which concurrently influence the measurements, pooling datasets poses unique challenges, e.g., variations in the data can come from both the acquisition method as well as the demographics of participants (gender, age). Invariant representation learning, by itself, is ill-suited to fully model the data generation process. In this paper, we show how bringing recent results on equivariant representation learning (for studying symmetries in neural networks) together with simple use of classical results on causal inference provides an effective practical solution to this problem. In particular, we demonstrate how our model allows dealing with more than one nuisance variable under some assumptions and can enable (relatively) painless analysis of pooled scientific datasets in scenarios that would otherwise entail removing a large portion of the samples.

## arXiv
Link to arXiv is https://arxiv.org/abs/2203.15234 

## Code
Available in the directory `code/`.

## Other Project particulars
The slides are available in the main directory with the title `slides_cvpr22.pdf`. We have a video going over the slides on youtube at this link https://www.youtube.com/watch?v=IxkyVbasi5s. 

A poster summarizing the project is in the main directory titled `poster_cvpr22.pdf`.


