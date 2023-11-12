import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
EMRE KARATAŞ
22001641
CS 464 
HOMEWORK 01
"""


def load_data():
    X_train = pd.read_csv('X_train.csv', sep=' ', header=0)
    X_test = pd.read_csv('X_test.csv', sep=' ', header=0)
    y_train = pd.read_csv('y_train.csv', header=None)
    y_test = pd.read_csv('y_test.csv', header=None)
    return X_train, X_test, y_train, y_test


def plot_class_distribution(y_train, y_test):
    train_percentages = y_train.iloc[:, 0].value_counts(normalize=True) * 100
    test_percentages = y_test.iloc[:, 0].value_counts(normalize=True) * 100

    # Plotting pie charts for both train and test datasets
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Pie chart for y_train
    axes[0].pie(train_percentages, labels=train_percentages.index, autopct='%1.1f%%', startangle=90, shadow=True)
    axes[0].set_title('Class Distribution in Training Set')

    # Pie chart for y_test
    axes[1].pie(test_percentages, labels=test_percentages.index, autopct='%1.1f%%', startangle=90, shadow=True)
    axes[1].set_title('Class Distribution in Test Set')

    # legend
    axes[0].legend(train_percentages.index, title="Classes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    # Adjusting the layout
    plt.tight_layout()
    plt.show()


def calculate_prior_probabilities(y_train):
    total_train_samples = y_train.shape[0]
    prior_probabilities_train = y_train.iloc[:, 0].value_counts() / total_train_samples
    return prior_probabilities_train


def word_occurrences_in_tech(X_train, y_train):
    tech_indices = y_train[y_train.iloc[:, 0] == 4].index
    tech_documents = X_train.iloc[tech_indices]

    # Replacing 'alien' and 'thunder' with the actual words from the header of x_train.csv
    alien_count = tech_documents['alien'].sum()
    thunder_count = tech_documents['thunder'].sum()

    # Calculate the total number of words in 'Tech' documents
    total_words_tech = tech_documents.sum().sum()

    # Calculate probabilities with Laplace smoothing
    vocab_size = len(X_train.columns)  # Total number of words in the vocabulary
    P_alien_given_tech = (alien_count + 1) / (total_words_tech + vocab_size)
    P_thunder_given_tech = (thunder_count + 1) / (total_words_tech + vocab_size)

    ln_P_alien_given_tech = np.log(P_alien_given_tech)
    ln_P_thunder_given_tech = np.log(P_thunder_given_tech)

    return alien_count, thunder_count, ln_P_alien_given_tech, ln_P_thunder_given_tech


def train_multinomial_nb(X, y):
    # Calculate the prior probabilities of each class
    class_priors = y.value_counts(normalize=True)
    # Calculate the likelihoods without smoothing
    word_counts = X.groupby(y.iloc[:, 0]).apply(lambda x: x.sum())
    word_likelihoods = word_counts.div(word_counts.sum(axis=1), axis=0)
    return class_priors, word_likelihoods


def train_multinomial_nb_with_smoothing(X, y, alpha=1):
    # Calculate the prior probabilities of each class
    class_priors = y.value_counts(normalize=True)

    # Calculate the word counts for each class and add alpha for smoothing
    word_counts = X.groupby(y.iloc[:, 0]).apply(lambda x: x.sum()) + alpha

    # Calculate the total word counts for each class (including the smoothing)
    total_word_counts = word_counts.sum(axis=1) + alpha * X.shape[1]

    # Calculate the likelihoods by dividing the word counts by the total word counts
    word_likelihoods = word_counts.div(total_word_counts, axis=0)

    return class_priors, word_likelihoods



def predict_multinomial_nb(X, class_priors, word_likelihoods):
    # Ensure all values are numeric for the dot product operation
    log_likelihoods = np.log(word_likelihoods).replace(-np.inf, -1e12).values
    log_priors = np.log(class_priors).replace(-np.inf, -1e12).values

    # Calculate log posterior for each class
    log_posterior = (X.values @ log_likelihoods.T) + log_priors

    # Return the class with the highest log posterior
    return np.argmax(log_posterior, axis=1)


def compute_confusion_matrix(y_true, y_pred, class_labels):
    # Initialize the confusion matrix to zeros
    confusion_matrix = pd.DataFrame(np.zeros((len(class_labels), len(class_labels))), index=class_labels,
                                    columns=class_labels)
    # Iterate over the true and predicted labels to increment the counts
    for true, pred in zip(y_true, y_pred):
        confusion_matrix.loc[true, pred] += 1

    return confusion_matrix


def main():
    X_train, X_test, y_train, y_test = load_data()

    while True:
        print("\n CS 464 Homework 01")
        print("\n Menu:")
        print("1. Print class counts for training and test sets")
        print("2. Plot class distribution")
        print("3. Calculate and display prior probabilities")
        print("4. Calculate and display word occurrences in 'Tech' class")
        print("5. Train a Multinomial Naive Bayes Model")
        print("6. Train a Multinomial Naive Bayes Model with Smoothing")
        print("7. Train a Bernoulli Naive Bayes Model")
        print("8. Exit")

        choice = input("Enter your choice (1-8): ")

        if choice == '1':
            print("Class counts in training set:")
            print(y_train.iloc[:, 0].value_counts())
            print("\nClass counts in test set:")
            print(y_test.iloc[:, 0].value_counts())

        elif choice == '2':
            plot_class_distribution(y_train, y_test)

        elif choice == '3':
            prior_probabilities_train = calculate_prior_probabilities(y_train)
            print("\nPrior probabilities in the training set:")
            print(prior_probabilities_train)

        elif choice == '4':
            alien_count, thunder_count, ln_P_alien_given_tech, ln_P_thunder_given_tech = word_occurrences_in_tech(
                X_train, y_train)
            print(f"\nAlien count in 'Tech': {alien_count}")
            print(f"Thunder count in 'Tech': {thunder_count}")
            print(f"Log probability of 'alien' given 'Tech': {ln_P_alien_given_tech}")
            print(f"Log probability of 'thunder' given 'Tech': {ln_P_thunder_given_tech}")

        elif choice == '5':
            # Train the Multinomial Naive Bayes model
            priors, likelihoods = train_multinomial_nb(X_train, y_train)

            # Predict on the test set
            y_pred = predict_multinomial_nb(X_test, priors, likelihoods)

            # Compute the class labels
            class_labels = sorted(y_train.iloc[:, 0].unique())

            # Compute the confusion matrix
            conf_matrix = compute_confusion_matrix(y_test.iloc[:, 0], y_pred, class_labels)
            # Calculate the accuracy
            accuracy = np.mean(y_pred == y_test.iloc[:, 0])

            print("Confusion Matrix:")
            print(conf_matrix)
            print(f"Accuracy: {accuracy:.3f}")
        elif choice == '6':
            # Train the model with additive smoothing
            class_priors, word_likelihoods = train_multinomial_nb_with_smoothing(X_train, y_train, alpha=1)

            # Predict on the test set
            y_pred = predict_multinomial_nb(X_test, class_priors, word_likelihoods)

            # Calculate accuracy
            accuracy = (y_test.iloc[:, 0] == y_pred).mean()

            # Generate the confusion matrix
            conf_matrix = compute_confusion_matrix(y_test.iloc[:, 0], y_pred, sorted(y_train.iloc[:, 0].unique()))

            # Print the accuracy and confusion matrix
            print("Confusion Matrix:")
            print(conf_matrix)
            print(f"Accuracy with additive smoothing: {accuracy:.3f}")

        elif choice == '7':

            # Converting the frequency data to binary
            X_train_binary = (X_train > 0).astype(int)
            X_test_binary = (X_test > 0).astype(int)

            # Initialize alpha for smoothing
            alpha = 1

            # Calculate prior probabilities
            class_priors = y_train.value_counts(normalize=True)

            # Calculate the likelihoods with additive smoothing
            word_counts = X_train_binary.groupby(y_train).apply(lambda x: x.sum() + alpha)
            total_counts = word_counts.sum(axis=1) + 2 * alpha
            word_likelihoods = word_counts.div(total_counts, axis=0)

            # Prediction function
            def predict(X, class_priors, word_likelihoods):
                # Calculate the log likelihoods
                log_likelihoods = np.log(word_likelihoods)
                log_likelihoods_not = np.log(1 - word_likelihoods)

                # Calculate total log likelihood for each class for the binary feature presence
                total_log_likelihood = X @ log_likelihoods.T + (1 - X) @ log_likelihoods_not.T

                # Add the log prior probabilities
                total_log_likelihood += np.log(class_priors)

                # Predict the class with the maximum log likelihood
                return total_log_likelihood.idxmax(axis=1)

            # Make predictions on the test set
            y_pred = predict(X_test_binary, class_priors, word_likelihoods)

            # Calculate accuracy
            accuracy = (y_pred == y_test).mean()

            # Confusion matrix
            conf_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

            # Display results
            print(f"Accuracy: {accuracy:.3f}")
            print("Confusion Matrix:")
            print(conf_matrix)

        elif choice == '8':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == '__main__':
    main()
