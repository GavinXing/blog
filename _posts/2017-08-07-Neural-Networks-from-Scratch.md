---
layout: post
comments: true
title:  "Neural Networks from Scratch"
excerpt: "We'll go through some basic ideas about NNs and RNNs, the intuitive and applications"
mathjax: true
tags:
  - deep learning
  - natural language processing
type: blog
---

Hi there, I'm a junior student from Shanghai JiaoTong University(SJTU). In my sophomore year, I started to learn about machine learning with a try of using *Support Vector Machine (SVM)* to classify credit card digits, something like the [MNIST TASK](http://yann.lecun.com/exdb/mnist/). Then I joined ADAPT and began my research work on NLP.

Within the NLP domain, many statistic methods work quite well, giving astonishing results on the basic tasks, *chunking* (which is especially important in Chinese), *Part of Speech tagging (POS)* and *Named-Entity recognition (NER)*. Models like *Hidden Markov Model (HMM)* and *Conditional random fields (CRF)* can give you a glimpse.

These days, I was attracted by the awesome performance of Deep Learning in NLP, which utilizes **Neural Networks** Models. I'll try my best to give an **overlook** over the Neural Networks stuff. On my own experience, the mathematic things sometimes distract me from the intuitive of how Neural Networks works. Thus, in this blog, there will only be some **"baby math"**, as [Prof. Chris Manning](https://nlp.stanford.edu/manning/) named in [CS224n](http://web.stanford.edu/class/cs224n/index.html).

Part of this blog overlaps with the [great blog](http://karpathy.github.io/neuralnets/) from Andrej Karpathy, which definitely worth reading. But I'll try to give you something more concrete about **backpropagation** by implement a linear regression model, and illustrate some **examples** and **applications** of Neural Networks in **NLP**, including word embedding and character-level language model.

Let's start!

## Chapter 1: Vanilla Neural Networks
At the first glance of the Neural Networks architecture, I wonder how it works so brilliantly. It seems non-sense that several layer of mathematic computation can simulate human brain, one of the most complicated creature in the world, even a little.

### Human Neurons
Let's turn to how our brain works to get an intuitive idea.

<div class="imgcap">
  <img src="/assets/rnn/neuron.png" style="border:none;">
  <div class="thecap">
    A brain neuron and its main components. Image credit: <a href="https://www.quora.com/What-is-an-intuitive-explanation-for-neural-networks">Quora</a>
  </div>
</div>

> Our brain has **a large network of interlinked neurons**, which act as a highway for information to be transmitted from point A to point B. When different information is sent from A to B, the brain **activates different sets of neurons**, and so essentially uses a different route to get from A to B.<br><br>
> At each neuron, dendrites receive incoming signals sent by other neurons. If the neuron **receives a high enough level of signals** within a certain period of time, the neuron sends an electrical pulse into the terminals. These **outgoing signals** are then received by other neurons.<br><br>
> Credit: [Quora answer by Annalyn Ng](https://www.quora.com/What-is-an-intuitive-explanation-for-neural-networks) (Quote part of her answer, click if you are interested, which I recommend you to do.)

### Modeling Human Neural Network
Let's recap what we can learn from last section:
- Information is sent between neurons.
- A neuron can be activated when received certain signal.

Here I'll introduce a vanilla Neural Networks Model, with a vector of length 8 as input, a hidden layer of 4 neurons, and a vector of length 3 as out put, the NN model can be trained as a image classifier for 3 tags.
<div class="imgcap">
  <img src="/assets/rnn/vanilla-nn-activated.png" style="border:none;">
  <div class="thecap">
    Vanilla Neural Network architecture, with one hidden layer
  </div>
  <div class="thecap">
    You can image that if you feed a picture involving animal into the network, it can output a tag of "cat" or "dog".
  </div>
  <div class="thecap">
    `W` and `b` are parameters of the model, which can be learned by training.
  </div>
</div>

As you can see in the picture, when  facing **an input of a cat picture**, the first and third (from left) **neurons are activated** (red), and then, these neurons send signal to the output layer and **get a "cat" output**.

Thinking about our brain do the same thing **when we see a cat**, some **neurons are triggered** and we **find out that it is a cat!**

This is the intuitive I see from the Neural Networks, though it is a simplified and idealized model just like most of the models in the world, it has the potential to improve machine intelligence.

> Practically, Neural Networks contains millions, even billions of *neurons*, each be sensitive to certain input, automatically learn features and representations, thus obtains intelligence.

Till now, I believe that you have obtained an intuitive idea about how Neural Networks works. Then I'll introduce how data flows from input to out, **a feedforward process**, and how the weight matrix and bias are trained, **a backpropagation process**.

### Feedforward
The feedforward process is straightforward, you can treat it as a function, feed it and it will give an output.

$$
y = NN(x)
$$

### Backpropagation
The backpropagation process is to tune the parameters, e.g. `W` and `b`, to fit the training data, minimize the total loss.
I'll take a simple linear regression model as an example, to illustrate how backpropagation works with Stochastic Gradient Descent (SGD).

Consider a network take a scalar `x` as input, and output `y = ax + b`. If we train the model on training set:

$$
X = [1, 2, 3, 4]
$$

$$
Y = [4, 3, 2, 1]
$$

The model will soon fit the function $$ y = -x + 5 $$, with *quadratic loss function* and *SGD*.

Let's take the derivative first!

$$
y = Wx + b ,\qquad
loss = \frac{1}{2} * {(y - y\_)}^{2}
$$

$$
\frac{dloss}{dy} = y - y\_ ,\qquad
\frac{dy}{dw} = x,\qquad \frac{dy}{db} = 1
$$

$$
\frac{dloss}{dw} = \frac{dloss}{dy} * \frac{dy}{dw} = (y - y\_) * x,\qquad
\frac{dloss}{db} = \frac{dloss}{dy} * \frac{dy}{db} = (y - y\_)
$$

> As you may acknowledge, **Chain Rule** is the key tool we should utilize. Backpropagation propagates through the chain rule from back to front.

Time to code! A toy implementation with pure Python:
``` python
X = [1, 2, 3, 4] # training set
Y = [4, 3, 2, 1]
w = 0 # initiate parameter as 0
b = 0

lr = 0.1 # learning rate
assert len(X) == len(Y)
for i in range(200):
    total_loss = 0
    for j in range(len(X)):
        x = X[j]
        y_ = Y[j]

        y = w * x + b # feed forward
        loss = (y - y_)**2 / 0.5
        total_loss += loss # accumulate loss

        dy = y - y_ # calculate derivative
        dw = dy * x
        db = dy * 1

        w -= lr * dw # backpropagation
        b -= lr * db
    print("After iteration {}, loss: {:.2f}. y = {:.2f}x + {:.2f}".format(i, total_loss, w, b))
```
Output:
```python
After iteration 0, loss: 44.47. y = -0.10x + 0.34
After iteration 1, loss: 39.21. y = -0.16x + 0.68
After iteration 2, loss: 33.80. y = -0.22x + 0.99
After iteration 3, loss: 29.14. y = -0.28x + 1.27
After iteration 4, loss: 25.12. y = -0.33x + 1.54
After iteration 5, loss: 21.66. y = -0.38x + 1.79
After iteration 6, loss: 18.67. y = -0.42x + 2.02
After iteration 7, loss: 16.10. y = -0.46x + 2.23
After iteration 8, loss: 13.88. y = -0.50x + 2.43
After iteration 9, loss: 11.96. y = -0.54x + 2.61
After iteration 10, loss: 10.31. y = -0.57x + 2.78
...
...
After iteration 51, loss: 0.02. y = -0.98x + 4.89
After iteration 52, loss: 0.02. y = -0.98x + 4.90
After iteration 53, loss: 0.02. y = -0.98x + 4.91
After iteration 54, loss: 0.02. y = -0.98x + 4.92
After iteration 55, loss: 0.01. y = -0.98x + 4.92
After iteration 56, loss: 0.01. y = -0.99x + 4.93
After iteration 57, loss: 0.01. y = -0.99x + 4.93
After iteration 58, loss: 0.01. y = -0.99x + 4.94
After iteration 59, loss: 0.01. y = -0.99x + 4.94
After iteration 60, loss: 0.01. y = -0.99x + 4.95
...
...
```
Let's recap what we have learned:
- An intuitive idea of how Neural Networks models human brain and how it works
- Feedforward process acts like a simple *function*
- Backpropagation utilizes **Chain Rule** to calculate derivative of each parameter, updates the parameters and minimize the loss

### Application: Word Vector
It is hard for us to encode words so that computer can use and meanwhile, keep the "meaning". A common way to utilize the "meaning" is to build a synonym set or hypernyms (is-a) relationship set, like *WordNet*. It's useful but largely limited, limited to the relation set and the vocabulary set.

If we regard words as atomic symbols, we can use *one-hot representation* to encode all the words. Such representation also suffers lack of "meaning" and flexibility (influence all words when adding new words), and memory usage (13M words in Google News 1T corpora). The inner production of 2 different word vector is always `0`, means nothing.

However, researchers came up with an idea that we can get the meaning of a word by its neighbors.

> "You shall know a word by the company it keeps"  (J. R. Firth 1957: 11)

It reminds me of the years I was new to English. As a non-native speaker, I always look up in a dictionary when meeting a unknown word, but teachers said that "Never look up at first glance! *Guess its meaning from the context first!*".

Here's an example about word and context from CS224n:
> government debt problems turning into **banking** crises as has happened in <br>
> &emsp;&emsp;&nbsp;saying that Europe needs unified **banking** regulation to replace the hodgepodge

The words in the context represent *"banking"* !

Recent years, with the proposition of [Mikolov et al.](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf), distributed representation of word can be trained fast and successfully captures word's semantic meaning.Till now, there are 3 main models to train word vectors:
1. [word2vec](https://code.google.com/archive/p/word2vec/)
2. [GloVe](https://nlp.stanford.edu/projects/glove/)
3. [FastText](https://github.com/facebookresearch/fastText)

I'll introduce the word2vec model, for these 3 models are almost the same. GloVe takes statistic feature into account, and FastText predicts tag instead of words. I won't go detail to the training tricks, such as negative sampling, hierarchical-softmax, to name just a few.

The key idea of word2vec is:

> Predict between **every word** and its **context words**.

So obviously there are two algorithms:
1. **Skip-Gram (SG)**: Predict context words given target word
2. **Continuous-Bag-Of-Words (CBOW)**: Predict target word given context words

<div class="imgcap">
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" id="processonSvg1000" viewBox="130.5 233.0 773.5 433.0" width="773.5" height="433.0"><defs id="ProcessOnDefs1001"/><g id="ProcessOnG1002"><path id="ProcessOnPath1003" d="M130.5 233.0H904.0V666.0H130.5V233.0Z" fill="none"/><g id="ProcessOnG1004"><g id="ProcessOnG1005" transform="matrix(1.0,0.0,0.0,1.0,169.0,379.0)" opacity="1.0"><path id="ProcessOnPath1006" d="M0.0 0.0L307.0 0.0L307.0 36.0L0.0 36.0Z" stroke="#323232" stroke-width="2.0" stroke-dasharray="none" opacity="1.0" fill="#ffffff"/></g><g id="ProcessOnG1007" transform="matrix(1.0,0.0,0.0,1.0,169.0,460.0)" opacity="1.0"><path id="ProcessOnPath1008" d="M0.0 0.0L307.0 0.0L307.0 36.0L0.0 36.0Z" stroke="#323232" stroke-width="2.0" stroke-dasharray="none" opacity="1.0" fill="#ffffff"/></g><g id="ProcessOnG1009" transform="matrix(1.0,0.0,0.0,1.0,150.5,537.0)" opacity="1.0"><path id="ProcessOnPath1010" d="M0.0 0.0L86.0 0.0L86.0 31.0L0.0 31.0Z" stroke="none" stroke-width="0.0" stroke-dasharray="none" opacity="1.0" fill="none"/><g id="ProcessOnG1011" transform="matrix(1.0,0.0,0.0,1.0,0.0,4.25)"><text id="ProcessOnText1012" fill="#000000" font-size="18" x="42.0" y="18.45" font-family="微软雅黑" font-weight="normal" font-style="normal" text-decoration="none" family="微软雅黑" text-anchor="middle" size="18">selling</text></g></g><g id="ProcessOnG1013" transform="matrix(1.0,0.0,0.0,1.0,236.5,537.0)" opacity="1.0"><path id="ProcessOnPath1014" d="M0.0 0.0L86.0 0.0L86.0 31.0L0.0 31.0Z" stroke="none" stroke-width="0.0" stroke-dasharray="none" opacity="1.0" fill="none"/><g id="ProcessOnG1015" transform="matrix(1.0,0.0,0.0,1.0,0.0,4.25)"><text id="ProcessOnText1016" fill="#000000" font-size="18" x="42.0" y="18.45" font-family="微软雅黑" font-weight="normal" font-style="normal" text-decoration="none" family="微软雅黑" text-anchor="middle" size="18">these</text></g></g><g id="ProcessOnG1017" transform="matrix(1.0,0.0,0.0,1.0,322.5,537.0)" opacity="1.0"><path id="ProcessOnPath1018" d="M0.0 0.0L86.0 0.0L86.0 31.0L0.0 31.0Z" stroke="none" stroke-width="0.0" stroke-dasharray="none" opacity="1.0" fill="none"/><g id="ProcessOnG1019" transform="matrix(1.0,0.0,0.0,1.0,0.0,4.25)"><text id="ProcessOnText1020" fill="#000000" font-size="18" x="42.0" y="18.45" font-family="微软雅黑" font-weight="normal" font-style="normal" text-decoration="none" family="微软雅黑" text-anchor="middle" size="18">leather</text></g></g><g id="ProcessOnG1021" transform="matrix(1.0,0.0,0.0,1.0,408.5,537.0)" opacity="1.0"><path id="ProcessOnPath1022" d="M0.0 0.0L86.0 0.0L86.0 31.0L0.0 31.0Z" stroke="none" stroke-width="0.0" stroke-dasharray="none" opacity="1.0" fill="none"/><g id="ProcessOnG1023" transform="matrix(1.0,0.0,0.0,1.0,0.0,4.25)"><text id="ProcessOnText1024" fill="#000000" font-size="18" x="42.0" y="18.45" font-family="微软雅黑" font-weight="normal" font-style="normal" text-decoration="none" family="微软雅黑" text-anchor="middle" size="18">jackets</text></g></g><g id="ProcessOnG1025" transform="matrix(1.0,0.0,0.0,1.0,312.5,415.0)" opacity="1.0"><path id="ProcessOnPath1026" d="M10.0 0.0L20.0 10.0L13.4 10.0L13.4 45.0L6.6000000000000005 45.0L6.6000000000000005 10.0L0.0 10.0L10.0 0.0Z" stroke="#323232" stroke-width="2.0" stroke-dasharray="none" opacity="1.0" fill="#ffffff"/></g><g id="ProcessOnG1027" transform="matrix(1.0,0.0,0.0,1.0,183.5,496.0)" opacity="1.0"><path id="ProcessOnPath1028" d="M10.0 0.0L20.0 10.0L13.4 10.0L13.4 45.0L6.6000000000000005 45.0L6.6000000000000005 10.0L0.0 10.0L10.0 0.0Z" stroke="#323232" stroke-width="2.0" stroke-dasharray="none" opacity="1.0" fill="#ffffff"/></g><g id="ProcessOnG1029" transform="matrix(1.0,0.0,0.0,1.0,269.5,496.0)" opacity="1.0"><path id="ProcessOnPath1030" d="M10.0 0.0L20.0 10.0L13.4 10.0L13.4 45.0L6.6000000000000005 45.0L6.6000000000000005 10.0L0.0 10.0L10.0 0.0Z" stroke="#323232" stroke-width="2.0" stroke-dasharray="none" opacity="1.0" fill="#ffffff"/></g><g id="ProcessOnG1031" transform="matrix(1.0,0.0,0.0,1.0,355.5,496.0)" opacity="1.0"><path id="ProcessOnPath1032" d="M10.0 0.0L20.0 10.0L13.4 10.0L13.4 45.0L6.6000000000000005 45.0L6.6000000000000005 10.0L0.0 10.0L10.0 0.0Z" stroke="#323232" stroke-width="2.0" stroke-dasharray="none" opacity="1.0" fill="#ffffff"/></g><g id="ProcessOnG1033" transform="matrix(1.0,0.0,0.0,1.0,441.5,496.0)" opacity="1.0"><path id="ProcessOnPath1034" d="M10.0 0.0L20.0 10.0L13.4 10.0L13.4 45.0L6.6000000000000005 45.0L6.6000000000000005 10.0L0.0 10.0L10.0 0.0Z" stroke="#323232" stroke-width="2.0" stroke-dasharray="none" opacity="1.0" fill="#ffffff"/></g><g id="ProcessOnG1035" transform="matrix(1.0,0.0,0.0,1.0,312.5,334.0)" opacity="1.0"><path id="ProcessOnPath1036" d="M10.0 0.0L20.0 10.0L13.4 10.0L13.4 45.0L6.6000000000000005 45.0L6.6000000000000005 10.0L0.0 10.0L10.0 0.0Z" stroke="#323232" stroke-width="2.0" stroke-dasharray="none" opacity="1.0" fill="#ffffff"/></g><g id="ProcessOnG1037" transform="matrix(1.0,0.0,0.0,1.0,242.5,299.0)" opacity="1.0"><path id="ProcessOnPath1038" d="M0.0 0.0L160.0 0.0L160.0 40.0L0.0 40.0Z" stroke="none" stroke-width="0.0" stroke-dasharray="none" opacity="1.0" fill="none"/><g id="ProcessOnG1039" transform="matrix(1.0,0.0,0.0,1.0,0.0,8.125)"><text id="ProcessOnText1040" fill="#000000" font-size="19" x="79.0" y="19.475" font-family="微软雅黑" font-weight="normal" font-style="normal" text-decoration="none" family="微软雅黑" text-anchor="middle" size="19">fine</text></g></g><g id="ProcessOnG1041" transform="matrix(1.0,0.0,0.0,1.0,242.5,253.0)" opacity="1.0"><path id="ProcessOnPath1042" d="M0.0 0.0L160.0 0.0L160.0 40.0L0.0 40.0Z" stroke="none" stroke-width="0.0" stroke-dasharray="none" opacity="1.0" fill="none"/><g id="ProcessOnG1043" transform="matrix(1.0,0.0,0.0,1.0,0.0,7.5)"><text id="ProcessOnText1044" fill="#000000" font-size="20" x="79.0" y="20.5" font-family="微软雅黑" font-weight="normal" font-style="normal" text-decoration="none" family="微软雅黑" text-anchor="middle" size="20">CBOW</text></g></g><g id="ProcessOnG1045" transform="matrix(1.0,0.0,0.0,1.0,540.0,379.0)" opacity="1.0"><path id="ProcessOnPath1046" d="M0.0 0.0L72.0 0.0L72.0 36.0L0.0 36.0Z" stroke="#323232" stroke-width="2.0" stroke-dasharray="none" opacity="1.0" fill="#ffffff"/></g><g id="ProcessOnG1047" transform="matrix(1.0,0.0,0.0,1.0,628.0,379.0)" opacity="1.0"><path id="ProcessOnPath1048" d="M0.0 0.0L72.0 0.0L72.0 36.0L0.0 36.0Z" stroke="#323232" stroke-width="2.0" stroke-dasharray="none" opacity="1.0" fill="#ffffff"/></g><g id="ProcessOnG1049" transform="matrix(1.0,0.0,0.0,1.0,717.0,379.0)" opacity="1.0"><path id="ProcessOnPath1050" d="M0.0 0.0L72.0 0.0L72.0 36.0L0.0 36.0Z" stroke="#323232" stroke-width="2.0" stroke-dasharray="none" opacity="1.0" fill="#ffffff"/></g><g id="ProcessOnG1051" transform="matrix(1.0,0.0,0.0,1.0,805.0,379.0)" opacity="1.0"><path id="ProcessOnPath1052" d="M0.0 0.0L72.0 0.0L72.0 36.0L0.0 36.0Z" stroke="#323232" stroke-width="2.0" stroke-dasharray="none" opacity="1.0" fill="#ffffff"/></g><g id="ProcessOnG1053" transform="matrix(1.0,0.0,0.0,1.0,540.0,460.0)" opacity="1.0"><path id="ProcessOnPath1054" d="M0.0 0.0L72.0 0.0L72.0 36.0L0.0 36.0Z" stroke="#323232" stroke-width="2.0" stroke-dasharray="none" opacity="1.0" fill="#ffffff"/></g><g id="ProcessOnG1055" transform="matrix(1.0,0.0,0.0,1.0,628.0,460.0)" opacity="1.0"><path id="ProcessOnPath1056" d="M0.0 0.0L72.0 0.0L72.0 36.0L0.0 36.0Z" stroke="#323232" stroke-width="2.0" stroke-dasharray="none" opacity="1.0" fill="#ffffff"/></g><g id="ProcessOnG1057" transform="matrix(1.0,0.0,0.0,1.0,717.0,460.0)" opacity="1.0"><path id="ProcessOnPath1058" d="M0.0 0.0L72.0 0.0L72.0 36.0L0.0 36.0Z" stroke="#323232" stroke-width="2.0" stroke-dasharray="none" opacity="1.0" fill="#ffffff"/></g><g id="ProcessOnG1059" transform="matrix(1.0,0.0,0.0,1.0,805.0,460.0)" opacity="1.0"><path id="ProcessOnPath1060" d="M0.0 0.0L72.0 0.0L72.0 36.0L0.0 36.0Z" stroke="#323232" stroke-width="2.0" stroke-dasharray="none" opacity="1.0" fill="#ffffff"/></g><g id="ProcessOnG1061" transform="matrix(1.0,0.0,0.0,1.0,566.0,415.0)" opacity="1.0"><path id="ProcessOnPath1062" d="M10.0 0.0L20.0 10.0L13.4 10.0L13.4 45.0L6.6000000000000005 45.0L6.6000000000000005 10.0L0.0 10.0L10.0 0.0Z" stroke="#323232" stroke-width="2.0" stroke-dasharray="none" opacity="1.0" fill="#ffffff"/></g><g id="ProcessOnG1063" transform="matrix(1.0,0.0,0.0,1.0,654.0,415.0)" opacity="1.0"><path id="ProcessOnPath1064" d="M10.0 0.0L20.0 10.0L13.4 10.0L13.4 45.0L6.6000000000000005 45.0L6.6000000000000005 10.0L0.0 10.0L10.0 0.0Z" stroke="#323232" stroke-width="2.0" stroke-dasharray="none" opacity="1.0" fill="#ffffff"/></g><g id="ProcessOnG1065" transform="matrix(1.0,0.0,0.0,1.0,743.0,415.0)" opacity="1.0"><path id="ProcessOnPath1066" d="M10.0 0.0L20.0 10.0L13.4 10.0L13.4 45.0L6.6000000000000005 45.0L6.6000000000000005 10.0L0.0 10.0L10.0 0.0Z" stroke="#323232" stroke-width="2.0" stroke-dasharray="none" opacity="1.0" fill="#ffffff"/></g><g id="ProcessOnG1067" transform="matrix(1.0,0.0,0.0,1.0,831.0,415.0)" opacity="1.0"><path id="ProcessOnPath1068" d="M10.0 0.0L20.0 10.0L13.4 10.0L13.4 45.0L6.6000000000000005 45.0L6.6000000000000005 10.0L0.0 10.0L10.0 0.0Z" stroke="#323232" stroke-width="2.0" stroke-dasharray="none" opacity="1.0" fill="#ffffff"/></g><g id="ProcessOnG1069" transform="matrix(1.0,0.0,0.0,1.0,533.0,537.0)" opacity="1.0"><path id="ProcessOnPath1070" d="M0.0 0.0L86.0 0.0L86.0 31.0L0.0 31.0Z" stroke="none" stroke-width="0.0" stroke-dasharray="none" opacity="1.0" fill="none"/><g id="ProcessOnG1071" transform="matrix(1.0,0.0,0.0,1.0,0.0,4.25)"><text id="ProcessOnText1072" fill="#000000" font-size="18" x="42.0" y="18.45" font-family="微软雅黑" font-weight="normal" font-style="normal" text-decoration="none" family="微软雅黑" text-anchor="middle" size="18">fine</text></g></g><g id="ProcessOnG1073" transform="matrix(1.0,0.0,0.0,1.0,619.0,537.0)" opacity="1.0"><path id="ProcessOnPath1074" d="M0.0 0.0L86.0 0.0L86.0 31.0L0.0 31.0Z" stroke="none" stroke-width="0.0" stroke-dasharray="none" opacity="1.0" fill="none"/><g id="ProcessOnG1075" transform="matrix(1.0,0.0,0.0,1.0,0.0,4.25)"><text id="ProcessOnText1076" fill="#000000" font-size="18" x="42.0" y="18.45" font-family="微软雅黑" font-weight="normal" font-style="normal" text-decoration="none" family="微软雅黑" text-anchor="middle" size="18">fine</text></g></g><g id="ProcessOnG1077" transform="matrix(1.0,0.0,0.0,1.0,710.0,537.0)" opacity="1.0"><path id="ProcessOnPath1078" d="M0.0 0.0L86.0 0.0L86.0 31.0L0.0 31.0Z" stroke="none" stroke-width="0.0" stroke-dasharray="none" opacity="1.0" fill="none"/><g id="ProcessOnG1079" transform="matrix(1.0,0.0,0.0,1.0,0.0,4.25)"><text id="ProcessOnText1080" fill="#000000" font-size="18" x="42.0" y="18.45" font-family="微软雅黑" font-weight="normal" font-style="normal" text-decoration="none" family="微软雅黑" text-anchor="middle" size="18">fine</text></g></g><g id="ProcessOnG1081" transform="matrix(1.0,0.0,0.0,1.0,798.0,537.0)" opacity="1.0"><path id="ProcessOnPath1082" d="M0.0 0.0L86.0 0.0L86.0 31.0L0.0 31.0Z" stroke="none" stroke-width="0.0" stroke-dasharray="none" opacity="1.0" fill="none"/><g id="ProcessOnG1083" transform="matrix(1.0,0.0,0.0,1.0,0.0,4.25)"><text id="ProcessOnText1084" fill="#000000" font-size="18" x="42.0" y="18.45" font-family="微软雅黑" font-weight="normal" font-style="normal" text-decoration="none" family="微软雅黑" text-anchor="middle" size="18">fine</text></g></g><g id="ProcessOnG1085" transform="matrix(1.0,0.0,0.0,1.0,566.0,496.0)" opacity="1.0"><path id="ProcessOnPath1086" d="M10.0 0.0L20.0 10.0L13.4 10.0L13.4 45.0L6.6000000000000005 45.0L6.6000000000000005 10.0L0.0 10.0L10.0 0.0Z" stroke="#323232" stroke-width="2.0" stroke-dasharray="none" opacity="1.0" fill="#ffffff"/></g><g id="ProcessOnG1087" transform="matrix(1.0,0.0,0.0,1.0,654.0,496.0)" opacity="1.0"><path id="ProcessOnPath1088" d="M10.0 0.0L20.0 10.0L13.4 10.0L13.4 45.0L6.6000000000000005 45.0L6.6000000000000005 10.0L0.0 10.0L10.0 0.0Z" stroke="#323232" stroke-width="2.0" stroke-dasharray="none" opacity="1.0" fill="#ffffff"/></g><g id="ProcessOnG1089" transform="matrix(1.0,0.0,0.0,1.0,743.0,496.0)" opacity="1.0"><path id="ProcessOnPath1090" d="M10.0 0.0L20.0 10.0L13.4 10.0L13.4 45.0L6.6000000000000005 45.0L6.6000000000000005 10.0L0.0 10.0L10.0 0.0Z" stroke="#323232" stroke-width="2.0" stroke-dasharray="none" opacity="1.0" fill="#ffffff"/></g><g id="ProcessOnG1091" transform="matrix(1.0,0.0,0.0,1.0,831.0,496.0)" opacity="1.0"><path id="ProcessOnPath1092" d="M10.0 0.0L20.0 10.0L13.4 10.0L13.4 45.0L6.6000000000000005 45.0L6.6000000000000005 10.0L0.0 10.0L10.0 0.0Z" stroke="#323232" stroke-width="2.0" stroke-dasharray="none" opacity="1.0" fill="#ffffff"/></g><g id="ProcessOnG1093" transform="matrix(1.0,0.0,0.0,1.0,533.0,303.0)" opacity="1.0"><path id="ProcessOnPath1094" d="M0.0 0.0L86.0 0.0L86.0 31.0L0.0 31.0Z" stroke="none" stroke-width="0.0" stroke-dasharray="none" opacity="1.0" fill="none"/><g id="ProcessOnG1095" transform="matrix(1.0,0.0,0.0,1.0,0.0,4.25)"><text id="ProcessOnText1096" fill="#000000" font-size="18" x="42.0" y="18.45" font-family="微软雅黑" font-weight="normal" font-style="normal" text-decoration="none" family="微软雅黑" text-anchor="middle" size="18">selling</text></g></g><g id="ProcessOnG1097" transform="matrix(1.0,0.0,0.0,1.0,619.0,303.0)" opacity="1.0"><path id="ProcessOnPath1098" d="M0.0 0.0L86.0 0.0L86.0 31.0L0.0 31.0Z" stroke="none" stroke-width="0.0" stroke-dasharray="none" opacity="1.0" fill="none"/><g id="ProcessOnG1099" transform="matrix(1.0,0.0,0.0,1.0,0.0,4.25)"><text id="ProcessOnText1100" fill="#000000" font-size="18" x="42.0" y="18.45" font-family="微软雅黑" font-weight="normal" font-style="normal" text-decoration="none" family="微软雅黑" text-anchor="middle" size="18">these</text></g></g><g id="ProcessOnG1101" transform="matrix(1.0,0.0,0.0,1.0,705.0,303.0)" opacity="1.0"><path id="ProcessOnPath1102" d="M0.0 0.0L86.0 0.0L86.0 31.0L0.0 31.0Z" stroke="none" stroke-width="0.0" stroke-dasharray="none" opacity="1.0" fill="none"/><g id="ProcessOnG1103" transform="matrix(1.0,0.0,0.0,1.0,0.0,4.25)"><text id="ProcessOnText1104" fill="#000000" font-size="18" x="42.0" y="18.45" font-family="微软雅黑" font-weight="normal" font-style="normal" text-decoration="none" family="微软雅黑" text-anchor="middle" size="18">leather</text></g></g><g id="ProcessOnG1105" transform="matrix(1.0,0.0,0.0,1.0,791.0,303.0)" opacity="1.0"><path id="ProcessOnPath1106" d="M0.0 0.0L86.0 0.0L86.0 31.0L0.0 31.0Z" stroke="none" stroke-width="0.0" stroke-dasharray="none" opacity="1.0" fill="none"/><g id="ProcessOnG1107" transform="matrix(1.0,0.0,0.0,1.0,0.0,4.25)"><text id="ProcessOnText1108" fill="#000000" font-size="18" x="42.0" y="18.45" font-family="微软雅黑" font-weight="normal" font-style="normal" text-decoration="none" family="微软雅黑" text-anchor="middle" size="18">jackets</text></g></g><g id="ProcessOnG1109" transform="matrix(1.0,0.0,0.0,1.0,566.0,334.0)" opacity="1.0"><path id="ProcessOnPath1110" d="M10.0 0.0L20.0 10.0L13.4 10.0L13.4 45.0L6.6000000000000005 45.0L6.6000000000000005 10.0L0.0 10.0L10.0 0.0Z" stroke="#323232" stroke-width="2.0" stroke-dasharray="none" opacity="1.0" fill="#ffffff"/></g><g id="ProcessOnG1111" transform="matrix(1.0,0.0,0.0,1.0,652.0,334.0)" opacity="1.0"><path id="ProcessOnPath1112" d="M10.0 0.0L20.0 10.0L13.4 10.0L13.4 45.0L6.6000000000000005 45.0L6.6000000000000005 10.0L0.0 10.0L10.0 0.0Z" stroke="#323232" stroke-width="2.0" stroke-dasharray="none" opacity="1.0" fill="#ffffff"/></g><g id="ProcessOnG1113" transform="matrix(1.0,0.0,0.0,1.0,743.0,334.0)" opacity="1.0"><path id="ProcessOnPath1114" d="M10.0 0.0L20.0 10.0L13.4 10.0L13.4 45.0L6.6000000000000005 45.0L6.6000000000000005 10.0L0.0 10.0L10.0 0.0Z" stroke="#323232" stroke-width="2.0" stroke-dasharray="none" opacity="1.0" fill="#ffffff"/></g><g id="ProcessOnG1115" transform="matrix(1.0,0.0,0.0,1.0,824.0,334.0)" opacity="1.0"><path id="ProcessOnPath1116" d="M10.0 0.0L20.0 10.0L13.4 10.0L13.4 45.0L6.6000000000000005 45.0L6.6000000000000005 10.0L0.0 10.0L10.0 0.0Z" stroke="#323232" stroke-width="2.0" stroke-dasharray="none" opacity="1.0" fill="#ffffff"/></g><g id="ProcessOnG1117" transform="matrix(1.0,0.0,0.0,1.0,624.0,253.0)" opacity="1.0"><path id="ProcessOnPath1118" d="M0.0 0.0L160.0 0.0L160.0 40.0L0.0 40.0Z" stroke="none" stroke-width="0.0" stroke-dasharray="none" opacity="1.0" fill="none"/><g id="ProcessOnG1119" transform="matrix(1.0,0.0,0.0,1.0,0.0,7.5)"><text id="ProcessOnText1120" fill="#000000" font-size="20" x="79.0" y="20.5" font-family="微软雅黑" font-weight="normal" font-style="normal" text-decoration="none" family="微软雅黑" text-anchor="middle" size="20">SKIP-GRAM</text></g></g><g id="ProcessOnG1121" transform="matrix(1.0,0.0,0.0,1.0,340.0,600.0)" opacity="1.0"><path id="ProcessOnPath1122" d="M0.0 0.0L365.0 0.0L365.0 46.0L0.0 46.0Z" stroke="none" stroke-width="0.0" stroke-dasharray="none" opacity="1.0" fill="none"/><g id="ProcessOnG1123" transform="matrix(1.0,0.0,0.0,1.0,0.0,11.75)"><text id="ProcessOnText1124" fill="#000000" font-size="18" x="181.5" y="18.45" font-family="微软雅黑" font-weight="normal" font-style="normal" text-decoration="none" family="微软雅黑" text-anchor="middle" size="18">I am selling these fine leather jackets</text></g></g></g></g></svg>
<div class="thecap">
  CBOW vs. SKIP-GRAM
</div>
</div>

As shown in the diagram, CBOW takes context word as input and predicts the target word, SG generates the context-target pairs and predicts one context word given the target word.

It's time to dive deep into the skip-gram model. The diagram below is a clear representation from CS224n.

<div class="imgcap">
  <img src="/assets/rnn/skip-gram.png" style="border:none">
  <div class="thecap">
    Skiip-Gram model
  </div>
  <div class="thecap">
    Image credit: CS224n
  </div>
</div>



Define the notation:
- $$V$$ vocabulary size
- $$d$$ dimension of word embedding
- $$w_\epsilon$$ one-hot representation of the word
- $$W$$ word embedding matrix
- $$V_c$$ word vector of a word
- $$p(x{\vert}c)$$ the probability of context word x given center/target word c

We take one-hot representation $$w_\epsilon$$ to encode a word in dictionary, then look up in the word embedding matrix $$W$$ for the representation $$V_c$$, using the dot production $$V_c = Ww_\epsilon$$. After that, another dot production $$W^{'}V_c$$ is used to calculate the hidden representation of output word, then utilize $$softmax$$ to get the `probability representation` of output word. In the training, we have the truth answer, so we can calculate the loss and then backprop to tune the model parameters ($$W, W^{'}$$).

Given a vector $$x = [x_0, x_1, ..., x_{n-1}]$$:

$$
softmax{(x)}_i = \frac {e^{x_i}} {\sum_{j} e^{x^j}}
$$

Well, a little more explanation of the diagram. The `3` vectors at the end of the model represents all the context words for one center/target word. It stands for the several context-target pair we generates from the training corpora, doesn't mean that we predicts several context words on one feedforward.

If you are interested in how to code a word2vec model, here is a [toy example](https://gist.github.com/GavinXing/9954ea846072e115bb07d9758892382c) with PyTorch.

## Chapter 2: Recurrent Neural Networks

TODO
