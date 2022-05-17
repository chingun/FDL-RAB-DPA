# Foundations of Deep Learning Robust Deep Learning 

As deep learning-based techniques achieve state-of-the-art performance on a wide variety of tasks such as image recognition, malware classification etc., they will be increasingly used for consequential decision making in the near future. Recently, security concerns have attracted considerable attention both from practical and theoretical research communities. It has become clear that deep neural networks (DNNs) are vulnerable against a broad range of attacks. Among these attacks, the most well-known and studied ones are \emph{adversarial examples}, where the ill-intentioned adversary finds small perturbations to the correctly classified inputs in order to fool the DNN so that it produces an incorrect prediction during the inference phase.

Another equally important and yet understudied types of attack are \emph{data poisoning} attacks, where the adversary manipulates samples during the training phase so that the learned model becomes corrupted. A specific subset of \emph{data poisoning} attacks, called \emph{backdoor} attacks, are of special interest. These attacks insert a \emph{backdoor} into DNN by adding small triggers to a subset of training instances to bias the trained model towards test instances with the same patterns. Backdoor attacks are especially stealthy since backdoored DNN models behave correctly on benign (clean) test data, making them particularly challenging to identify. 


## RAB + DPA
[Update: 05/15/2022] The architecture works with 4px poisoning attack on DPA same as RAB to run the entire architecture you need to run the following steps in succession.<br/>

1. Partition code: <br/>
```
python partition_data_norm_hash.py --dataset cifar --num_partitions 50
``` 
2. Train a base classifier on each partition example here 50 (you can choose any partition size and the base classifiers will be saved in checkpoints): <br/>
```
python train_cifar_nin_baseline.py --start_partition 0 --num_partitions 50
```
3. Create the score matrix from all test inputs based on the base classifiers: <br/>
```
python eval_cifar.py --models <base classifier folder created by train_cifar_nin_baseline.py>
```
4. Certify the final dpa classifier with the ultimate score: <br/>
```
python certify.py --evaluations <name of the eval file generated from step 3> --num_partitions 50
```

