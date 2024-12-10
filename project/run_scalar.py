"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import random

import minitorch


class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        """Initialize a Neural Network

        Here, we simply choose what layers to add to our NN.

        the forward() method has the logic for the forward pass of the network,
        and has listed 3 layers. Thus, we need to have 3 layers.
        """
        super().__init__()
        # TODO: Implement for Task 1.5.

        # 2 inputs, `hidden_layers` number of hidden layer neurons,
        # and 1 output (the final prediction)
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        middle = [h.relu() for h in self.layer1.forward(x)]
        end = [h.relu() for h in self.layer2.forward(middle)]
        return self.layer3.forward(end)[0].sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        """Initialize a Linear Layer

        Initialize a linear layer in a neural network. A linear layer
        is fully connected. That is, every input neuron is connected to
        every output neuron.

        For docs please see: https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html

        Logic of this function (this code was implemented by the instructors).

        For every input neuron, we create an empty list of weights, and then
        for every output neuron, we add a weight to the list. We do this
        because a linear layer is fully connected, and each connection
        has a weight.

        We also add a bias term for each output neuron. Thus, for every output
        neuron, we append a bias parameter (this bias parameter is the same
        for every output neuron in this layer).

        Args:
            in_size (_type_): the number of input neurons
            out_size (_type_): the number of output neurons
        """
        super().__init__()
        self.weights = []
        self.bias = []
        for i in range(in_size):
            self.weights.append([])
            for j in range(out_size):
                self.weights[i].append(
                    self.add_parameter(
                        f"weight_{i}_{j}", minitorch.Scalar(2 * (random.random() - 0.5))
                    )
                )
        for j in range(out_size):
            self.bias.append(
                self.add_parameter(
                    f"bias_{j}", minitorch.Scalar(2 * (random.random() - 0.5))
                )
            )

    def forward(self, inputs):
        """Forward step for a linear layer

        For a linear layer, we move forward by taking the dot product of the
        input and the weights, and then adding the bias,
        for each output neuron.

        That is, for each output neuron, we take the dot product of the input
        and the weights, and then add the bias.

        We return a list of the values for each output neuron.

        Args:
            inputs (_type_): _description_
        """
        # TODO: Implement for Task 1.5.
        y = [b.value for b in self.bias]
        for i, x in enumerate(inputs):
            for j in range(len(y)):
                y[j] = y[j] + x * self.weights[i][j].value
        return y


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class ScalarTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(self.hidden_layers)

    def run_one(self, x):
        return self.model.forward(
            (minitorch.Scalar(x[0], name="x_1"), minitorch.Scalar(x[1], name="x_2"))
        )

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            loss = 0
            for i in range(data.N):
                x_1, x_2 = data.X[i]
                y = data.y[i]
                x_1 = minitorch.Scalar(x_1)
                x_2 = minitorch.Scalar(x_2)
                out = self.model.forward((x_1, x_2))

                if y == 1:
                    prob = out
                    correct += 1 if out.data > 0.5 else 0
                else:
                    prob = -out + 1.0
                    correct += 1 if out.data < 0.5 else 0
                loss = -prob.log()
                (loss / data.N).backward()
                total_loss += loss.data

            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    # defualts
    PTS = 50
    HIDDEN = 2
    RATE = 0.5

    # PTS = 50
    # HIDDEN = 10
    # RATE = 0.001
    data = minitorch.datasets["Simple"](PTS)
    ScalarTrain(HIDDEN).train(data, RATE)
