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
        self.g3 = nn.Linear(32, k)   # Final encoder layer
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        # Decoder
        self.h1 = nn.Linear(k, 32)   # First decoder layer
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
    # TODO: Add a regularizer to the cost function.

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    # optimizer = optim.ASGD(model.parameters(), lr=lr)
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
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))

        epochs.append(epoch)
        train_losses.append(train_loss)
        val_accs.append(valid_acc)

    return epochs, train_losses, val_accs


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

        # Perform multiple forward passes with dropout enabled
        mc_outputs = torch.cat([model(inputs) for _ in range(num_mc_samples)], dim=0)

        # Average the outputs over Monte Carlo samples
        avg_output = torch.mean(mc_outputs, dim=0)

        guess = avg_output[valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def tune_and_plot(params):
    num_questions = params["train_matrix"].shape[1]
    train_data = params["train_matrix"]
    zero_train_matrix = params["zero_train_matrix"]
    valid_data = params["valid_data"]
    # num_epoch = params["num_epoch"]

    _, axs = plt.subplots(len(params["epochs_list"]), len(params["lrs_list"]), figsize=(18, 12))

    best_model_info = {"val_accuracy": -1}

    for i, epochs in enumerate(params["epochs_list"]):
        for j, lr in enumerate(params["lrs_list"]):
            for latent_dim in params["latent_dims"]:
                ae_model = DeepAutoEncoder(num_questions, latent_dim)
                epoch_ticks, _, val_accuracy = train(ae_model, lr, train_data,
                                                     zero_train_matrix, valid_data,
                                                     epochs)
                axs[i][j].plot(epoch_ticks, val_accuracy, label=f"Latent Dimension = {latent_dim}")
                axs[i][j].set_xticks(np.arange(0, epochs, epochs // 5))
                axs[i][j].set_ylabel("Validation Accuracy (%)")
                axs[i][j].set_title(f"Epochs: {epochs}, Learning Rate: {lr}", fontsize=10)

                # Update the best model info if the current model is better
                if max(val_accuracy) > best_model_info["val_accuracy"]:
                    best_model_info = {
                        "val_accuracy": max(val_accuracy),
                        "latent_dim": latent_dim,
                        "lr": lr,
                        "epochs": epochs,
                    }
            axs[i][j].legend(loc='upper left', prop={'size': 6})

    plt.savefig("tuned_results_partb.png")
    print("Best Model Info: ")
    print("Latent Dimension: ", best_model_info["latent_dim"])
    print("Learning Rate: ", best_model_info["lr"])
    print("Number of Epochs: ", best_model_info["epochs"])
    print("Validation Accuracy: ", best_model_info["val_accuracy"])
    return best_model_info["latent_dim"], best_model_info["lr"], best_model_info["epochs"]


def tune_lambda_and_plot(params):
    num_questions = params["train_matrix"].shape[1]
    train_data = params["train_matrix"]
    zero_train_matrix = params["zero_train_matrix"]
    valid_data = params["valid_data"]
    lr = params["best_lr"]
    num_epoch = params["best_epochs"]
    latent_dim = params["best_k"]

    fig, axs = plt.subplots(2, 1, figsize=(12, 12))  # Changed this line

    best_model_info = {"val_accuracy": -1, "train_loss": float("inf")}

    for i, lamb in enumerate(params["lambdas"]):
        ae_model = DeepAutoEncoder(num_questions, params["best_k"])
        epoch_ticks, train_loss, val_accuracy = train(ae_model, params["best_lr"], train_data,
                                                      zero_train_matrix, valid_data,
                                                      params["best_epochs"], lamb)
        axs[0].plot(epoch_ticks, val_accuracy, label=f"Lambda = {lamb}")  # Changed this line
        axs[1].plot(epoch_ticks, train_loss, label=f"Lambda = {lamb}")  # Added this line

        axs[0].set_xticks(np.arange(0, params["best_epochs"], params["best_epochs"] // 5))
        axs[0].set_ylabel("Validation Accuracy (%)")
        axs[1].set_xticks(
            np.arange(0, params["best_epochs"], params["best_epochs"] // 5))  # Added this line
        axs[1].set_ylabel("Train Loss")  # Added this line

        # Update the best model info if the current model is better
        if max(val_accuracy) > best_model_info["val_accuracy"]:
            best_model_info = {
                "val_accuracy": max(val_accuracy),
                "lambda": lamb,
                "train_loss": train_loss[-1]
            }
        axs[0].legend(loc='upper left', prop={'size': 6})
        axs[1].legend(loc='upper left', prop={'size': 6})  # Added this line

    plt.savefig("lambda_tuned_results_partb.png")

    # Train the model with the best lambda and report test accuracy
    best_model = DeepAutoEncoder(num_questions, latent_dim)
    _, _, _ = train(best_model, lr, train_data, zero_train_matrix,
                    valid_data, num_epoch, best_model_info["lambda"])
    test_accuracy = evaluate(best_model, zero_train_matrix, params["test_data"])
    print("Best Lambda Info: ")
    print("Lambda: ", best_model_info["lambda"])
    print("Validation Accuracy: ", best_model_info["val_accuracy"])
    print("Train Loss: ", best_model_info["train_loss"])  # Added this line
    print(f"Final test accuracy is: {test_accuracy}")


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    tune_params = {
        "train_matrix": train_matrix,
        "zero_train_matrix": zero_train_matrix,
        "valid_data": valid_data,
        "latent_dims": [32, 64, 128, 256, 512],
        "lrs_list": [0.001, 0.003, 0.005, 0.01],
        "epochs_list": [15, 22, 30, 37]
    }

    # best_k, lr, num_epoch = tune_and_plot(tune_params)

    # Select your best k (latent dimension)
    best_k = 128
    num_question = train_matrix.shape[1]

    # Create your model using the best k
    best_model = DeepAutoEncoder(num_question, best_k)
    # Define your learning rate and number of epochs
    lr = 0.003
    num_epoch = 30

    # Train the model and capture the training and validation history
    epochs, train_losses, val_accs = train(best_model, lr, train_matrix, zero_train_matrix,
                                           valid_data, num_epoch)

    # Evaluate on test data
    test_accuracy = evaluate(best_model, zero_train_matrix, test_data)
    print(f"Final test accuracy is: {test_accuracy}")

    # Plot training and validation loss curves over epochs
    plt.figure(figsize=(10, 5))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()

    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accs, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Q3E
    # tuning lambda
    tune_lambda_params = {
        "train_matrix": train_matrix,
        "zero_train_matrix": zero_train_matrix,
        "valid_data": valid_data,
        "lambdas": [0.001, 0.002, 0.003],
        "best_k": best_k,
        "best_lr": lr,
        "best_epochs": num_epoch,
        "test_data": test_data
    }

    tune_lambda_and_plot(tune_lambda_params)


if __name__ == "__main__":
    main()
