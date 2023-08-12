# TODO: complete this file.
from starter_code.part_a.item_response import *
# from mpl_toolkits.mplot3d import Axes3D


def bootstrap_sample(data, by_student=False):
    """
    Generates a bootstrap sample of the data.
    by_student: If True, samples students and their entire data. Otherwise, samples individual data points.
    """
    u_ids, q_ids, c_ids = np.array(data["user_id"]), np.array(data["question_id"]), np.array(
        data["is_correct"])

    if by_student:
        unique_students = np.unique(u_ids)
        sampled_students = np.random.choice(unique_students, len(unique_students), replace=True)

        mask = np.isin(u_ids, sampled_students)
        sampled_u, sampled_q, sampled_c = u_ids[mask], q_ids[mask], c_ids[mask]
    else:
        indices = np.random.choice(len(u_ids), len(u_ids), replace=True)
        sampled_u, sampled_q, sampled_c = u_ids[indices], q_ids[indices], c_ids[indices]

    return {"user_id": sampled_u, "question_id": sampled_q, "is_correct": sampled_c}


def average_predictions(data, thetas, betas):
    """
    Averages the predictions of models with parameters in thetas and betas.
    """
    total_predictions = np.zeros(len(data["is_correct"]))

    for t, b in zip(thetas, betas):
        model_prediction = irt_prediction(data, t, b)
        total_predictions += model_prediction

    return (total_predictions / len(thetas) >= 0.5).astype(int)


def irt_prediction(data, theta, beta):
    """
    Returns binary predictions for a model defined by theta and beta.
    """
    predictions = [sigmoid((theta[user] - beta[question]).sum()) for user, question in
                   zip(data["user_id"], data["question_id"])]
    return np.array(predictions)


def assess_accuracy(data, predictions):
    """
    Evaluates the accuracy of the predictions.
    """
    return np.mean(np.array(data["is_correct"]) == predictions)


def single_model_performance(learning_rate, num_iterations):
    # Load datasets
    train_data, val_data, test_data = load_train_csv("../data"), load_valid_csv(
        "../data"), load_public_test_csv("../data")

    # Train model on the entire training data
    theta, beta, _, _ = irt(train_data, val_data, learning_rate, num_iterations)

    final_train_accuracy = evaluate(train_data, theta, beta)
    final_val_accuracy = evaluate(val_data, theta, beta)
    final_test_accuracy = evaluate(test_data, theta, beta)

    # Return accuracies
    return (final_train_accuracy,
            final_val_accuracy,
            final_test_accuracy)


def ensemble_method(learning_rate=0.01, num_iterations=100):
    # Load datasets
    train_data, val_data, test_data = load_train_csv("../data"), load_valid_csv(
        "../data"), load_public_test_csv("../data")

    num_models = 3
    thetas, betas = [], []

    # Train models on bootstrap samples
    for _ in range(num_models):
        sample = bootstrap_sample(train_data)
        theta, beta, _, _ = irt(sample, val_data, learning_rate, num_iterations)
        thetas.append(theta)
        betas.append(beta)

    # Calculate ensemble predictions
    ensemble_train_preds = average_predictions(train_data, thetas, betas)
    ensemble_val_preds = average_predictions(val_data, thetas, betas)
    ensemble_test_preds = average_predictions(test_data, thetas, betas)

    # Return ensemble accuracies
    return (assess_accuracy(train_data, ensemble_train_preds),
            assess_accuracy(val_data, ensemble_val_preds),
            assess_accuracy(test_data, ensemble_test_preds))


def ensemble_method_with_tuning_and_plotting():
    # Define the hyperparameters to tune
    learning_rates = [0.003, 0.015, 0.1]
    num_iterations_list = [20, 50, 100]

    best_val_accuracy = 0
    best_hyperparams = (None, None)

    # Store accuracies for plotting
    train_accuracies, val_accuracies = [], []

    # Grid search
    for lr in learning_rates:
        for num_iterations in num_iterations_list:
            print(f"Training with learning rate: {lr} and iterations: {num_iterations}")

            train_accuracy, val_accuracy, _ = ensemble_method(lr, num_iterations)

            # Append accuracies for plotting
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            # Check if this combo of hyperparams is the best
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_hyperparams = (lr, num_iterations)

            print(
                f"Training accuracy with learning rate {lr} and iterations {num_iterations}: {train_accuracy}")
            print(
                f"Validation accuracy with learning rate {lr} and iterations {num_iterations}: {val_accuracy}")

    best_train_accuracy, best_val_accuracy, best_test_accuracy = ensemble_method(*best_hyperparams)

    single_train_accuracy, single_val_accuracy, single_test_accuracy = single_model_performance(
        *best_hyperparams)

    # Plotting
    fig, ax = plt.subplots()

    # For each learning rate, plot how accuracy changes with the number of iterations
    for i, lr in enumerate(learning_rates):
        subset_train_accuracies = train_accuracies[i * len(num_iterations_list):(i + 1) * len(
            num_iterations_list)]
        subset_val_accuracies = val_accuracies[
                                i * len(num_iterations_list):(i + 1) * len(num_iterations_list)]

        ax.plot(num_iterations_list, subset_train_accuracies, marker='o',
                label=f'Training Accuracy (LR={lr})')
        ax.plot(num_iterations_list, subset_val_accuracies, marker='x', linestyle='--',
                label=f'Validation Accuracy (LR={lr})')

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracies')
    ax.legend()

    plt.show()

    print(
        f"Best validation accuracy is {best_val_accuracy} with learning rate {best_hyperparams[0]} and iterations {best_hyperparams[1]}")
    print(f"Test accuracy with best hyperparameters: {best_test_accuracy}")

    print(f"Single Model Train Accuracy: {single_train_accuracy}")
    print(f"Single Model Validation Accuracy: {single_val_accuracy}")
    print(f"Single Model Test Accuracy: {single_test_accuracy}")

    print(f"Ensemble Train Accuracy: {best_train_accuracy}")
    print(f"Ensemble Validation Accuracy: {best_val_accuracy}")
    print(f"Ensemble Test Accuracy with best hyperparameters: {best_test_accuracy}")

    if best_test_accuracy > single_test_accuracy:
        print("The ensemble method improved test performance!")
    else:
        print("The single model performed better or equally on the test set.")


if __name__ == "__main__":
    ensemble_method_with_tuning_and_plotting()
