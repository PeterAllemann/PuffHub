# 2d - MLP

Analysis of the results

For this task we used DeepDIVA. We trained a MLP on both the MNIST and the Permutated MNIST dataset with 1 hidden layer.
We chose the parameters, for which the performance in task 2b was the best: 
* lr = 0.1
* decay-lr by factor 10 after every 5 epochs
* training epochs = 20
* hidden layer = 100  

 We trained the model with 3 different initializations, like in task 2b. 

| Dataset| mean Accuracy [%]| 
| -------------|------------|
| MNIST |97.813| 
| Permutated MNIST | 97.873 |


Test accuracy on the permutated MNIST dataset is almost the same as on the regular MNIST dataset (± 0.060%), see table above.
Accuracy on the training set and on the validation set is also almost the same as on the permutated MNIST dataset. On both datasets the accuracy is very high (>97%).

There is almost no difference because in MLP every neuron from one layer is connected with every neuron of the other layer, so the order of input does not matter here.




