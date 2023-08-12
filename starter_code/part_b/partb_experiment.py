from starter_code.utils import *
from torch.autograd import Variable

import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch
import matplotlib.pyplot as plt


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class DeepAutoEncoder(nn.Module):
    def __init__(self, num_question, k=100, dropout_p=0.5):
        super(DeepAutoEncoder, self).__init__()

        # Encoder
        self.g1 = nn.Linear(num_question, 64)  # Reduced to 64 nodes
        self.g2 = nn.Linear(64, 32)  # Added another encoder layer
        self.g3 = nn.Linear(32, k)  # Final encoder layer
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        # Decoder
        self.h1 = nn.Linear(k, 32)  # First decoder layer
        self.h2 = nn.Linear(32, 64)  # Added another decoder layer
        self.h3 = nn.Linear(64, num_question)  # Final decoder layer

        # Dropout
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, inputs):
        # Encoder
        out = self.g1(inputs)
        # out = torch.relu(out)  # Switched to ReLU
        out = self.leaky_relu(out)
        out = self.dropout(out)

        out = self.g2(out)
        # out = torch.relu(out)  # Switched to ReLU
        out = self.leaky_relu(out)
        out = self.dropout(out)

        out = self.g3(out)
        # out = torch.relu(out)  # Switched to ReLU
        out = self.leaky_relu(out)
        out = self.dropout(out)

        # Decoder
        out = self.h1(out)
        # out = torch.relu(out)  # Switched to ReLU
        out = self.leaky_relu(out)
        out = self.dropout(out)

        out = self.h2(out)
        # out = torch.relu(out)  # Switched to ReLU
        out = self.leaky_relu(out)
        out = self.dropout(out)

        out = torch.sigmoid(self.h3(out))
        return out

    def get_weight_norm(self):
        g1_w_norm = torch.norm(self.g1.weight, 2) ** 2
        g2_w_norm = torch.norm(self.g2.weight, 2) ** 2
        g3_w_norm = torch.norm(self.g3.weight, 2) ** 2
        h1_w_norm = torch.norm(self.h1.weight, 2) ** 2
        h2_w_norm = torch.norm(self.h2.weight, 2) ** 2
        h3_w_norm = torch.norm(self.h3.weight, 2) ** 2
        return g1_w_norm + g2_w_norm + g3_w_norm + h1_w_norm + h2_w_norm + h3_w_norm


def train(model, lr, train_data, zero_train_data, valid_data, num_epoch, lamb=0):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.Adagrad(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    train_losses = []
    val_accs = []
    epochs = []

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)

            # Adding regularization
            loss += (lamb/2) * model.get_weight_norm()

            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        # print("Epoch: {} \tTraining Cost: {:.6f}\t "
        #       "Valid Acc: {}".format(epoch, train_loss, valid_acc))

        epochs.append(epoch)
        train_losses.append(train_loss)
        val_accs.append(valid_acc)

    return epochs, train_losses, val_accs
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data, num_mc_samples=10):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :param num_mc_samples: integer representing number of monte carlo samples
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        # output = model(inputs)

        # # Print the current user ID and question ID being evaluated
        # print("Current User ID:", u)
        # print("Current Question ID:", valid_data["question_id"][i])

        # Perform multiple forward passes with dropout enabled
        mc_outputs = torch.cat([model(inputs) for _ in range(num_mc_samples)], dim=0)

        # # Print the shape of mc_outputs
        # print("MC Outputs Shape:", mc_outputs.shape)

        # Average the outputs over Monte Carlo samples
        avg_output = torch.mean(mc_outputs, dim=0)

        # # Print the shape of avg_output
        # print("Avg Output Shape:", avg_output.shape)
        #
        # # Print the valid question ID
        # print("Valid Question ID:", valid_data["question_id"][i])

        guess = avg_output[valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def experiment(k, lr, num_epoch, mc_num_sample_lst, dropout_prob_lst, train_matrix,
               zero_train_matrix, valid_data, test_data, lamb):
    test_accuracy_lst = []
    for num in mc_num_sample_lst:
        for p in dropout_prob_lst:
            num_question = train_matrix.shape[1]
            model = DeepAutoEncoder(num_question, k, p)
            _, _, _ = train(model, lr, train_matrix, zero_train_matrix,
                                                   valid_data, num_epoch, lamb)
            test_accuracy = evaluate(model, zero_train_matrix, test_data, num)
            test_accuracy_lst.append(test_accuracy)
            print(f"MC Samples: {num}, Dropout Prob: {p}, Test Accuracy: {test_accuracy}")

            # Reset the model for the next iteration
            model = DeepAutoEncoder(num_question, k, p)

def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    best_k = 128

    # Define your learning rate and number of epochs
    lr = 0.003
    num_epoch = 30
    mc_num_sample_lst = [5, 10, 20]
    dropout_prob_lst = [0.0, 0.25, 0.5, 0.75]

    # Experiment
    experiment(best_k, lr, num_epoch, mc_num_sample_lst, dropout_prob_lst, train_matrix,
               zero_train_matrix, valid_data, test_data, 0.002)


if __name__ == "__main__":
    main()
