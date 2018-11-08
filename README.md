# FashionMNIST

CNN based on the architecture and implementation by ashmeet13 [1]
to attain high accuracy on Fashion MNIST [2]
and subsequently visualize the information plane [3]

This model is comprised of the following layers:
    - 2 conv layers + Max Pooling + Batch Normalisation (Using ReLU activation)
    - 2 Fully Connected Layers

    Random Horizontal Flips of the Fashion MNIST images is included as a data augmentation technique.

    Optimizer Used - Adam

Final Accuracy - 93.29%

Notes on the information plane visualization:
    - Mutual information between the labels and the hidden layers were originally computed empirically, but [4], which uses a k nearest neighbors estimator, yielded slightly more reasonable mutual information estimates
    - To visualize the information plane, follow the instructions below

Visualizing the information plane
Run `Fashion.py` to generate the appropriate `.dat` files containing the cached mutual information scores (updated every epoch (or more precisely, every 300 minibatches, because the term epoch loses some meaning when random data augmentation is used))
```
python3 Fashion.py
```
then run the cells in `information_plane.ipynb` to generate the plot.

References
[1] https://github.com/ashmeet13/FashionMNIST-CNN
[2] https://github.com/zalandoresearch/fashion-mnist
[3] https://arxiv.org/abs/1703.00810
[4] https://github.com/gregversteeg/NPEET
