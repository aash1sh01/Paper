## Project: Adversarial Attack on Neural Network Based Models

### Abstract
Neural Networks have become an integral part of computer software recently. Ever since Alexnet (Krizhevsky et al. 2012) won the Imagenet competition with an end to end Convolutional Neural Network, deep learning models have been used widely in applications ranging from search engines to self driving cars. As a result, it is being increasingly important to make sure these models work correctly and ensure they cannot be attacked. Covulational Neural Networks(CNN) have shown near human accuracy on image classification problems. However, recent research have shown that many of such models are vulnerable to attacks. We discuss a small subset of such attacks and demonstrate them in our paper.  CNN(s) are widely used in facial recognition algorithms. Are Neural Nets as reliable as the human cortex? We answer this question with extensive research and demo of attacking a pre trained NN that initially predicts images correctly and watch it fail.

### Introduction
Majority of the research behind adversarial attack on neural network started as a side-effect of when researching an interesting property of neural networks. Szegedy et al. 2013 tried to learn what vision models were learning about images and found that a small change to images (small enough to be imperceptible to a human) was enough to fool the model to output much different predictions. See image below:

<img src="https://miro.medium.com/max/1838/1*PmCgcjO3sr3CPPaCpy5Fgw.png"> <p style="text-align:center;font-style:italic">Image from Towards Data Science Medium </p>

This is an example of perturbation attack which is the first attack discovered and perhaps the most popular type of attack. Throughout this paper, we will only focus on this type of attack. Since the finding, numerous other ways of fooling models have been researched. Especially as a result of wide use of neural networks in critical applications like self-driving cars, it could quite literally be a life-saver to study and prevent attacks against such attacks. We can easily imagine a scenario where a model gives false predictions, a stop sign to be something else or a car ahead as a stop sign.

### Types of Attacks
There are several types of attacks but we can categorize them based on the knowledge we have about the model we are trying to attack. In brief, following are the types of adversarial attacks on neural network.

1. **Black Box Attack**
	Black box attacks assume that the attackers do not have any information about the model that the attackers are trying to attack. This is most of the case in a real-world scenario where for example an attacker might try to fool Tesla's self driving model or some authentication model. In such cases, attackers do not have any or have very little information about the architecture or the parameters of the model that they are trying to attack and as a result is harder.
2. **White Box Attack**
	White box attack assume that the attacker have information about the model that they wish to attack. This is not always the case in real-world scenario but allows attacker to attack such models much easily due to the specific knowledge about the model.

Although we categorize these attacks as totally different, there has been research by Goodfellow et al. 2014 that has shown that attacks are transferable i.e. a attack on equivalent model can be transferred to another model. So, a white box attack on a known model can be transferred as a black box attack on an equivalent model which blurs the lines between our categories.

### Formal Definition
A typical DNN can be presented as the function $F : X → Y$, which maps from an input set $X$ to a label set $Y$ . $Y$ is a set of $k$ classes like ${1, 2, . . . , k}$. For a sample $x ∈ X$, it is correctly classified by $F$ to the truth label $y$, i.e., $F(x) = y$.

An attacker aims at adding a small perturbation $ε$ in $x$ to create adversarial example $x'$, such that $F(x') = y'(y \ne y')$. 

### Attack on Vision Model using Fast Gradient Sign Method
We have used Fast Gradient Sign method to attack on a pretrained Image classifier model that initially classifies all the animals and recognizes their species. The idea is to add a small perturbation not recognizable to the human eyes that could potentially change the predictions of the model. FGSM involves taking cost function gradient with respect to each input feature of the image to the original image.  
Let $\theta$ be the parameters or the weights of the neural network, $X$ the input to the network  and $Y$ the target classes we train the network with. Now, we denote the cost  function of our neural network as $J(\theta,X,Y)$. To generate a sample with applied perturbation, i.e the adversarial sample $X_{adv}$, the fast gradient sign method uses the below rule: 

$$x_{adv}= X+ ε.sign(\nabla_{X}J(\theta,X,Y))$$

The below image shows how we change the cost function gradient to apply small perturbation epsilon that changes the machine's prediction of a Macau's image. 

<img src="  
https://i.imgur.com/LA9ykWm.png"/>
<p style= "text-align:center;font-style:italic">A macau bird predicted to be a bookcase.</p>

The below image shows the working of your model in an abstract level:

<img src="https://i.imgur.com/sgx0s27.png"/>
<p style= "text-align:center;font-style:italic">Schematic diagram of how DNN(s) can easily be fooled.</p>



The psuedocode of the algorithm we used to perform Fast Gradient Sign attack on our pretrained model.:

<img src="https://i.imgur.com/OzzK6iK.png"/>

*Here **M** is the model **X** is the input, **Y** is the set of labels and **epsilon** is the degree of perturbation applied.*



### Our attack and it's effects



Applying to the above formula, we have performed a similar attack ourselves and the results are not surprising.  We take the $\epsilon$ value to be as low as 0.04 and attack our model with the Fast Gradient Sign method. The findings of our algorithms are described with images as folows:



<img src="  
https://i.imgur.com/eckexV9.png"/>
<p style= "text-align:center;font-style:italic">A frog predicted as a  bird after performing our attack.</p>

<img src="https://i.imgur.com/RCzWeZh.png"/>

<p style= "text-align:center;font-style:italic">Two different breeds of dogs both predicted to be a horse.</p>


<img src="https://i.imgur.com/PW7nmhV.png"/>
<p style= "text-align:center;font-style:italic">A car predicted as a horse.</p>



In the above examples, we have seen our perfect model malfunction and mispredict on various instances. In first example, we saw that two completely unrelated species a frog and a bird confusing our neural network. A rather peculiar behavior was absorbed in the second example, two different breeds of dogs were identified as horses in both the instances. In the third example, we can see that inanimate objects as random as an automobile is being identified as a living animal, a horse. These different examples give us a sense of how differently the neural networkds work than the human cortex. We can see that even after being perfect when trained with the original images a small perturbation(not concievable to human eyes) was enough for the neural networks to completely mispredict. We have also analysed that these attacks completely dismantle the neural networks. Two completely unrelated things like a car and a horse, a car being predicted as a horse is a very unintelligent thing and the whole existence of AI is deemed questionable.


### Risks
1. We argue that if attackers attack on a very secure intelligence facility's security system that is accessible only to certain faces, they can breach the security and enter by attacking on the model. The facial recognition algorithm used by the facility can be attacked and malicious people can access the facility.
2. We also think that self-driving cars fueled with AI can be easily attacked and manipulated. The stop sign can be misread as a different sign causing accidents and various misconducts.
3. We also believe that with so many utilities and applications using DNN and Covulated Neural Networks to predict/classify human behaviours from social media to highly secure intelligence facilities, they being tampered with can cause loss of millions of dollars and might even cause cyber wars between nations.

### Prevention

1. To prevent from being attacked with adversarial attacks, the first and foremost intuition is to train the models with possible adversarial images that can be used by the attackers to attack the model. 
2. The model architecture and model weights must be kept as a high level secret not accessible to the attackers.



### Conclusion
We have seen and observed a perfect cutting-edge model fail with as less as changing the gradient with some noise. We have analysed and assesed the risk of this stretching to national security. We have also proposed and verified different ways that can be taken to make sure these attacks do not take place or to mitigate the effects of these attacks. To conclude, we can see that the importance of machine learning algorithms is increasing day by day with each application and utility using those and to make them safe and to prevent different possible attacks is of grave concern as the world now is swiftly transistioning to automation and AI.
### References
- Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." _Advances in neural information processing systems_ 25 (2012): 1097-1105.
- Szegedy, Christian, et al. "Intriguing properties of neural networks." _arXiv preprint arXiv:1312.6199_ (2013).
- Goodfellow, Ian J., Jonathon Shlens, and Christian Szegedy. "Explaining and harnessing adversarial examples." _arXiv preprint arXiv:1412.6572_ (2014).
- Both, Rawlani, Chibandry. "A tutorial on attacking DNNs with Adversarial Examples" (2017).
