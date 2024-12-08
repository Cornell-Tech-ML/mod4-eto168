"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch


# Use this function to make a random parameter in
# your module.
def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)


# TODO: Implement for Task 2.5.


class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        """Initialize a Neural Network

        Here, we simply choose what layers to add to our NN.

        the forward() method has the logic for the forward pass of the network,
        and has listed 3 layers. Thus, we need to have 3 layers.
        """
        super().__init__()

        # 2 inputs, `hidden_layers` number of hidden layer neurons,
        # and 1 output (the final prediction)
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        """
        Forward for a Tensor-based Neural Network

        Here, we define the activation functions for each layer,
        then specify the final layer, which uses a sigmoid to classify.
        """
        # as the linear layers have been defined, we can now use them
        # same as scalar. Forward layers, with an activation function
        # final layer is a sigmoid to predict
        h = self.layer1.forward(x).relu()
        h = self.layer2.forward(h).relu()
        return self.layer3.forward(h).sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size) -> None:
        """
        Initialize Linear Layer
        """
        super().__init__()

        # initialize random parameter objects for weights and bias
        # weights_shape = [in_size, out_size]
        # bias_shape = out_size
        # weights_data = RParam(in_size, out_size)
        # bias_data = RParam(bias_shape)

        # self.weights = minitorch.Tensor(
        #     minitorch.TensorData(weights_data, shape=weights_shape)
        # )
        # self.bias = minitorch.Tensor(minitorch.TensorData(bias_data, shape=bias_shape))
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        """Use tensor operations to update weights"""
        # # now we need to approach training from a tensor perspective.
        # # for a linear layer, we need to multiply the input by the weights
        # # and add the bias. This is the forward pass. We cannot use for
        # # loops. Notice that in TensorTrain, run_one and run_many
        # # pass in minitorch.tensor([x]) and minitorch.tensor(X) respectively.
        # # this means that we are working with a tensor.

        # # we need to multiply the input by the weights
        # # we can use mul_zip to multiply the input by the weights
        # # using broadcast

        # # because RParam returns a Parameter object, we need to access the
        # # value
        # weights_values = self.weights.value
        # bias_values = self.bias.value

        # # now, we a linear layer is multiplcation with
        # # the weights and addition with the bias
        # # Without reshaping the tensors, we get
        # # IndexingError: Shapes (16, 2) and (2, 2) are not broadcastable.
        # # for inputs * weights_values
        # # thus, we need to reshape with view.
        # # Using broadcasting rules:
        # # we line up:
        # #
        # # 16 2
        # # 2  2
        # #
        # # So, we need (16 2 1) and (1 2 2) to be able to broadcast

        # dim1, dim2 = inputs.shape
        # inputs_broadcastable = inputs.view(dim1, dim2, 1)
        # weights_broadcastable = weights_values.view(1, dim2, self.out_size)
        # bias_broadcastable = bias_values.view(1, self.out_size)

        # # now, we multiply the inputs by the weights, then sum them
        # # along the 1st dimension, reshape, then add the bias
        # x = (inputs_broadcastable * weights_broadcastable).sum(1).view(
        #     dim1, self.out_size
        # ) + bias_broadcastable
        # return x
        batch, in_size = x.shape
        return self.weights.value.view(1, in_size, self.out_size) * x.view(
            batch, in_size, 1
        ).sum(1).view(batch, self.out_size) + self.bias.value.view(self.out_size)


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
