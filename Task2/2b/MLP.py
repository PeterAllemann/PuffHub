

import torch.nn as nn


class Flatten(nn.Module):
    """
    Flatten a convolution block into a simple vector.

    Replaces the flattening line (view) often found into forward() methods of networks. This makes it
    easier to navigate the network with introspection
    """
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class PR_MLP(nn.Module):


    def __init__(self, output_channels=10, **kwargs):
        super(PR_MLP, self).__init__()

        self.expected_input_size = (28,28)

        hidden = 20
	
        if 'hidden_size' in kwargs: 
            hidden = kwargs['hidden_size']
	
	# First layer
        self.fc1 = nn.Sequential(
            Flatten(),
            nn.Linear((28*28*3),  hidden),
	    nn.ReLU()
        )

        # Classification layer
        self.fc2 = nn.Sequential(	
            nn.Linear(hidden, output_channels)
        )


    def forward(self, x):


        """
        Computes forward pass on the network

        Parameters
        ----------
        x : Variable
            Sample to run forward pass on. (input to the model)

        Returns
        -------
        Variable
            Activations of the fully connected layer
        """
        x = self.fc1(x)
        x = self.fc2(x)
        return x
