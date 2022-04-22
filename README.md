# Foundations of Deep Learning Robust Deep Learning 

As deep learning-based techniques achieve state-of-the-art performance on a wide variety of tasks such as image recognition, malware classification etc., they will be increasingly used for consequential decision making in the near future. Recently, security concerns have attracted considerable attention both from practical and theoretical research communities. It has become clear that deep neural networks (DNNs) are vulnerable against a broad range of attacks. Among these attacks, the most well-known and studied ones are \emph{adversarial examples}, where the ill-intentioned adversary finds small perturbations to the correctly classified inputs in order to fool the DNN so that it produces an incorrect prediction during the inference phase.

Another equally important and yet understudied types of attack are \emph{data poisoning} attacks, where the adversary manipulates samples during the training phase so that the learned model becomes corrupted. A specific subset of \emph{data poisoning} attacks, called \emph{backdoor} attacks, are of special interest. These attacks insert a \emph{backdoor} into DNN by adding small triggers to a subset of training instances to bias the trained model towards test instances with the same patterns. Backdoor attacks are especially stealthy since backdoored DNN models behave correctly on benign (clean) test data, making them particularly challenging to identify. 