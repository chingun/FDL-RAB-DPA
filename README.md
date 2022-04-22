# Foundations of Deep Learning Robust Deep Learning 

As deep learning-based techniques achieve state-of-the-art performance on a wide variety of tasks such as image recognition, malware classification etc., they will be increasingly used for consequential decision making in the near future. Recently, security concerns have attracted considerable attention both from practical and theoretical research communities. It has become clear that deep neural networks (DNNs) are vulnerable against a broad range of attacks. Among these attacks, the most well-known and studied ones are \emph{adversarial examples}, where the ill-intentioned adversary finds small perturbations to the correctly classified inputs in order to fool the DNN so that it produces an incorrect prediction during the inference phase.

Another equally important and yet understudied types of attack are \emph{data poisoning} attacks, where the adversary manipulates samples during the training phase so that the learned model becomes corrupted. A specific subset of \emph{data poisoning} attacks, called \emph{backdoor} attacks, are of special interest. These attacks insert a \emph{backdoor} into DNN by adding small triggers to a subset of training instances to bias the trained model towards test instances with the same patterns. Backdoor attacks are especially stealthy since backdoored DNN models behave correctly on benign (clean) test data, making them particularly challenging to identify. 


## RAB
Randomized smoothing is a method for constructing a new smoothed classifier $g(x)$ from a  base classifier $f(x)$, where $x$ is the input itself. The smoothed classifier returns whichever class the base classifier $f$ is most likely to return when $x$ is perturbed by a certain type of noise (e.g. Gaussian).  Cohen et al. \cite{cohen} apply the randomized smoothing technique to defend the model against evasion attacks. They also provide a tight robustness guarantee for their randomized smoothing method. Based on this work, Weber et al. \cite{weber} have recently proposed techniques to defend the model against both evasion and backdoor attacks, under a unified framework, via randomized smoothing and hypothesis testing. They also derive a tight robustness bound and conduct extensive scale experiments to evaluate their certified robustness against backdoor attacks.

## DPA
Deep Partition Aggregation (DPA) is proposed by Levine and Feizi \cite{levine}. This method defends against general and label-flipping poisoning attacks. DPA partitions the training set into ``$k$" partitions, where a hash function determines the partition assignment for a training sample. Since the hash value depends only on the value of the training sample, neither poisoning other samples, nor changing the total number of samples, nor reordering the samples can change what the partition is assigned to. DPA trains $k$ base classifiers separately, one on each partition. The test step evaluates each classifier on the test sample and returns the plurality classification as the final result. The key insight is that removing a training sample or adding a new sample will only change the contents of one partition and therefore, will only affect the classification of one of the $k$ base classifiers. This leads to robustness certifications against general poisoning attacks. Authors in \cite{levine} claim that their results are the current state-of-the-art provable defenses against general data poisoning attacks.

## RAB + DPA
[04/22/2022] Currently we are observing DPA performance with RAB type of poisoning attacks. Accuracy measurements available via the following code: <br/>
```
python train_cifar_nin_baseline.py --start_partition 0 --num_partitions 50
```
