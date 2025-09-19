import numpy as np
from statistics import mean, stdev

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score



def NNModel(sub_df, df_squamosa):
    # Split the data into training, validation, and testing sets
    x_train, x_temp, y_train, y_temp = train_test_split(sub_df, df_squamosa, test_size=0.3, random_state=1)
    x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.5, random_state=1)

    # Define a function to create a neural network architecture
    def create_model(input_dim, output_dim, num_layers=2, units= 8, learning_rate=0.001):
        model = Sequential()
        for _ in range(num_layers):
            model.add(Dense(units, activation='relu', input_dim=input_dim))

        model.add(Dense(output_dim, activation='softmax'))
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model

    # Define hyperparameter search space
    param_space = {
        'num_layers': [2, 3, 4],
        'units': [16, 32, 64],
        'learning_rate': [0.00001, 0.0001, 0.001]
    }

    #carrot
    param_space = {
        'num_layers': [2, 3, 4, 5],
        'units': [8, 16, 32, 64, 128, 256],
        'learning_rate': [0.0001, 0.001, 0.01, 0.1]
    }

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # Random search for hyperparameters
    best_model = None
    best_accuracy = 0
    best_roc_auc = 0
    best_f1_score = 0
    best_history = None
    best_params = None
    acc_list = []
    roc_auc_list = []
    f1_list = []

    for _ in range(10):  # Number of random trials
        params = {param: np.random.choice(values) for param, values in param_space.items()}
        model = create_model(x_train.shape[1], len(np.unique(y_train)), **params)
        
        history = model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val), callbacks=[early_stopping], verbose=0)
        
        #y_pred = np.argmax(model.predict(x_test), axis=1)
        #accuracy = accuracy_score(y_test, y_pred)
        #roc_auc = roc_auc_score(y_test, y_pred)
        #f1 = f1_score(y_test, y_pred)
        #acc_list.append(float(accuracy))
        #roc_auc_list.append(float(roc_auc))
        #f1_list.append(float(f1))

        y_probs = model.predict(x_test)

        if y_probs.shape[1] == 2:  # Softmax output for binary/multiclass
            y_score = y_probs[:, 1]  # for ROC AUC (binary)
        else:  # Sigmoid output
            y_score = y_probs.squeeze()

        y_pred = np.argmax(y_probs, axis=1)

        accuracy = accuracy_score(y_test, y_pred)

        # If binary classification, y_test should be binary
        if len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, y_score)
        else:
            roc_auc = roc_auc_score(y_test, y_probs)

        f1 = f1_score(y_test, y_pred)

        acc_list.append(float(accuracy))
        roc_auc_list.append(float(roc_auc))
        f1_list.append(float(f1))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_roc_auc = roc_auc
            best_f1_score = f1
            best_history = history
            best_params = params

    print(f"Best Accuracy: {best_accuracy}")
    plt.plot(best_history.history['loss'], label='Train Loss')
    plt.plot(best_history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()

    plt.plot(best_history.history['accuracy'], label='Train Accuracy')
    plt.plot(best_history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    plt.show()

    mean_acc = mean(acc_list) * 100
    std_acc = stdev(acc_list) * 100
    mean_roc_auc = mean(roc_auc_list)
    std_roc_auc = stdev(roc_auc_list)
    mean_f1 = mean(f1_list)
    std_f1 = stdev(f1_list)

    return mean_acc, std_acc, mean_roc_auc, std_roc_auc, mean_f1, std_f1