from starter_code.utils import *

import numpy as np
import matplotlib.pyplot as plt
import random


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    for i, c in enumerate(data["is_correct"]):
        u = data["user_id"][i]
        q = data["question_id"][i]
        x = theta[u] - beta[q]
        log_lklihood += c * x - np.log(1 + np.exp(x))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    user_id_arr = np.array(data["user_id"])
    question_id_arr = np.array(data["question_id"])
    correct_arr = np.array(data["is_correct"])

    theta_copy = theta.copy()
    beta_copy = beta.copy()
    for i in range(len(theta)):
        theta[i] -= lr * (
                np.sum(
                    sigmoid(theta_copy[i] - beta_copy)[question_id_arr[user_id_arr == i]]) - np.sum(
                        correct_arr[user_id_arr == i]))

    for j in range(len(beta)):
        beta[j] -= lr * (np.sum(correct_arr[question_id_arr == j]) - np.sum(
            sigmoid(theta_copy - beta_copy[j])[user_id_arr[question_id_arr == j]]))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(542)
    beta = np.zeros(1774)

    val_acc, train_acc = [], []
    val_log_likelihood, train_log_likelihood = [], []

    for _ in range(iterations):
        train_neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_log_likelihood.append(train_neg_lld)

        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        val_log_likelihood.append(val_neg_lld)

        train_score = evaluate(data=data, theta=theta, beta=beta)
        train_acc.append(train_score)

        val_score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc.append(val_score)
        print(
            "NLLK: {} \t Train Score: {} \t Validation Score: {}".format(train_neg_lld, train_score,
                                                                         val_score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_log_likelihood, train_log_likelihood


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 0.003
    iterations = 100

    theta, beta, val_log_likelihood, train_log_likelihood = irt(train_data, val_data, lr,
                                                                iterations)

    plt.plot(train_log_likelihood, label="Training NLLK", color='blue')
    plt.plot(val_log_likelihood, label="Validation Accuracy", color='red')
    plt.xlabel("Iteration")
    plt.ylabel("NLLK Value")
    plt.legend()
    plt.show()

    final_val_accuracy = evaluate(data=val_data, theta=theta, beta=beta)
    final_test_accuracy = evaluate(data=test_data, theta=theta, beta=beta)

    print("Validation Accuracy: {:.2f}%".format(final_val_accuracy * 100))
    print("Test Accuracy: {:.2f}%".format(final_test_accuracy * 100))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    questions = random.sample([i for i in range(1774)], 3)

    # Plot the probability of correct response for the selected questions
    theta_range = np.linspace(min(theta), max(theta), 1000)
    for question_id in questions:
        beta_value = beta[question_id]
        probabilities = [sigmoid(t - beta_value) for t in theta_range]
        plt.plot(theta_range, probabilities, label=f'Question {question_id}')

    plt.xlabel('Theta')
    plt.ylabel('P(c_ij = 1)')
    plt.legend()
    plt.title('Probability of Correct Response as a Function of Theta')
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
