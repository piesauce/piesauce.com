---
layout: page
title: Projects
---

## Research Topics

### Neural Style Transfer for text.
The meaning of a text is a composition of the content of the text and the style or tone in which it is written. Stylistic dimensions include formality, politeness, diplomacy, understandability, persuasiveness, etc. Style transfer for text is the task of taking a text expressed in a particular style and converting it into a text expressing a different style. 

Generally, there is a tradeoff between the coherence of the translated text and the degree to which the style has been transferred. This is because the more edits are made to the original text, the more danger of it losing its original meaning. My research focuses on developing techniques to achieve transfer strength while preserving the original meaning of the text.

My focus is on unsupervised methods that can perform this task. Current techniques being experimented include encoder-decoder models with pointer networks, disambiguating between content and style in the latent space using VAE’s, and adversarial training. Heavy inspiration is taken from the closely related task of machine translation.

I am also focused on developing reliable evaluation metrics that can measure the coherence, meaning preservation, and style transfer success of the generated text.

### Abusive Language Detection:
Abusive language takes on several forms - hate speech, cyberbullying, trolling, grooming, etc. Detection of abusive language is treated as a text classification task, closely related to sentiment analysis. I am working on developing architectures that can facilitate robust models that are able to perform multi-level classification in the face of lexical alterations made to the text to evade abusive language filters. My current focus is on sub-word representations and transfer learning.


### Adversarial attacks on NLP systems
NLP systems are vulnerable from a security and privacy standpoint. Due to the lack of robustness in neural models, a small adversarial perturbation to the input text can cause the model to misclassify it. My current research focuses on generating proof-of-concept attacks with imperceptible perturbations that preserve the meaning of the original text, as well as defenses against these attacks. 

I am also interested in privacy-preserving NLP systems, specifically in terms of preventing demographic information leakage from hidden layer activations of neural networks.

#### Publications

1. Pai, S [”Automated Data Classification for mainframes”](https://research.tue.nl/en/studentTheses/automated-data-classification-for-mainframes), Master’s thesis, Eindhoven
University of Technology, 2012
2. Pai, S ; Sharma, Y et al ”Formal Verification of OAuth 2.0 using the Alloy
framework” In proc. of International Conference of Communications Systems and
Network Technologies, Jammu, 2011

