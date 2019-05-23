# 2b - MLP

For this task we used DeepDIVA. We trained MLP with one hidden layer to classify handwritten digits (MNIST dataset).
To optimize the performance (in terms of classification accuracy), we played around with different hyperparameters:
* learning rate:
	* (0.001, 0.003, 0.01, 0.03, 0.1)
* hidden layer size (number of neurons) 
	* (10, 30, 50, 100, 500)
* epochs (number of training iterations) 
	* (5, 10, 20)
	
For each parameter combination, we trained the model with 3 different initializations. 

The best accuracy (on the test set) was 98.22%, with:
* learning rate = 0.1
* hidden layer size = 100
* epochs = 20

We observed that the bigger the hidden layer size and the epochs, the higher is the accuracy.

The worst performance was observed with hidden layer size = 10, epochs = 5 and learning rate = 0.001. 

We also observed that if the learning rate is small (e.g. 0.001) we have to train longer.  



