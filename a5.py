import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("alzheimers_disease_data.csv")

# Drop PatientID since it's just an identifier and DoctorInCharge since it's not relevant and same for all
df.drop(columns=["PatientID"], inplace=True)
df.drop(columns=["DoctorInCharge"], inplace=True)

# Separate features (all columns - diagnosis) and target(diagnosis)
X = df.drop(columns=["Diagnosis"])  # Features
y = df["Diagnosis"]  # Target (Alzheimer's diagnosis: 0 = No, 1 = Yes)

# ✅ Convert y to ensure it's numeric (some datasets store 0/1 as strings)
y = y.astype(int)  # Ensure y is integer type

# --- 1️⃣ One-Hot Encoding for Categorical Features ---
categorical_features = ["Ethnicity", "EducationLevel"]
encoder = OneHotEncoder(sparse_output=False)  # Drop first to avoid redundancy
X_encoded = encoder.fit_transform(df[categorical_features])

# Convert encoded data to DataFrame
encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_features))

# Drop original categorical columns & concatenate encoded ones
X = X.drop(columns=categorical_features)
X = pd.concat([X, encoded_df], axis=1)

# --- 2️⃣ Standardization (Z-score normalization) for Clinical Features ---
standard_features = [
    "BMI", "SystolicBP", "DiastolicBP", "CholesterolTotal",
    "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides",
    "MMSE", "ADL", "FunctionalAssessment"
]
scaler_standard = StandardScaler()
X[standard_features] = scaler_standard.fit_transform(X[standard_features])

# --- 3️⃣ Min-Max Scaling for Lifestyle Features ---
minmax_features = ["DietQuality", "SleepQuality", "PhysicalActivity", "AlcoholConsumption"]
scaler_minmax = MinMaxScaler()
X[minmax_features] = scaler_minmax.fit_transform(X[minmax_features])

print("✅ Data preprocessing complete! Preprocessed files saved.")
#print the fist ten lines of the processed data
#print("First ten lines of the processed data:")
#print(X.head(10))

# Create output directory for saving results
def create_output_directory():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"A5_{timestamp}"

    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    return output_dir

# --- Define a function to build the Neural Network with SGD optimizer ---
def build_model(hidden_neurons, hidden_neurons_2nd_layer, hidden_neurons_3rd_layer):
    model = keras.Sequential([
        layers.Dense(hidden_neurons, activation="elu", kernel_regularizer=regularizers.l2(0.001)),  # Hidden layer
        layers.Dense(hidden_neurons_2nd_layer, activation="elu", kernel_regularizer=regularizers.l2(0.001)),  # 2nd Hidden layer
        layers.Dense(hidden_neurons_3rd_layer, activation="elu", kernel_regularizer=regularizers.l2(0.001)),  # 3rd Hidden layer
        layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))  # Output layer
    ])

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.6),
                  loss="binary_crossentropy",
                  metrics=['accuracy', 'binary_crossentropy', 'mse'])
    return model

# --- Early Stopping Callback ---
early_stopping_val_loss = EarlyStopping(
    monitor='val_loss',
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
    verbose=1,
    mode='min'
)

i=X.shape[1]  # Number of features
hidden_neurons = [i//2, 2*i//3, i, 2*i]  # Hidden neurons for the second layer
hidden_neurons_2nd_layer = [i//2, 2*i//3, i, 2*i]  # Hidden neurons for the first layer
hidden_neurons_3rd_layer = [i//2, 2*i//3, i, 2*i]

# --- Function to evaluate hyperparameter combinations ---
def evaluate_hyperparameters(X, y, hidden_neurons, hidden_neurons_2nd_layer, hidden_neurons_3rd_layer):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize dictionaries to store results
    results = {
        'hidden_neurons': [],
        'hidden_neurons_2nd_layer': [],
        'hidden_neurons_3rd_layer': [],
        'val_accuracy': [],
        'val_loss': [],
        'val_binary_crossentropy': [],
        'val_mse': [],
        'epochs_trained': [],
        'train_accuracy': [],
        'generalization_gap': [],  # To track the generalization gap
        'all_histories': []  # Store all training histories
    }

    # Evaluate each hyperparameter combination
    for hidden_neuron in hidden_neurons:
        for hidden_neuron_2nd_layer in hidden_neurons_2nd_layer:
            for hidden_neuron_3rd_layer in hidden_neurons_3rd_layer:
                print(f"\nEvaluating {hidden_neuron} hidden neurons in the first hidden layer")
                print(f" and {hidden_neuron_2nd_layer} hidden neurons in the secodn hidden layer")
                print(f" and {hidden_neuron_3rd_layer} hidden neurons in the third hidden layer")

                fold_accuracies = []
                fold_losses = []
                fold_bce = []
                fold_mse = []
                fold_epochs = []
                all_fold_histories = []
                val_scores = []
                fold_train_acc = []

                # Perform k-fold cross-validation
                for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
                    print(f"  Fold {fold + 1}/5")
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    # Build model with current hyperparameters
                    model = build_model(hidden_neuron, hidden_neuron_2nd_layer, hidden_neuron_3rd_layer)

                    # Train the model
                    history = model.fit(
                        X_train, y_train,
                        epochs=100,
                        batch_size=32,
                        verbose=0,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping_val_loss]
                    )

                    # Record training history
                    all_fold_histories.append(history.history)

                    # Evaluate model performance
                    val_scores = model.evaluate(X_test, y_test, verbose=0)
                    train_scores = model.evaluate(X_train, y_train, verbose=0)
                    val_loss = val_scores[0]  # loss (MSE)
                    val_accuracy = val_scores[1]  # accuracy
                    val_bce = val_scores[2]  # binary crossentropy
                    val_mse = val_scores[3]  # MSE
                    epochs = len(history.history['loss'])

                    # Store individual fold results
                    fold_accuracies.append(val_accuracy)
                    fold_losses.append(val_loss)
                    fold_bce.append(val_bce)
                    fold_mse.append(val_mse)
                    fold_epochs.append(epochs)
                    fold_train_acc.append(train_scores[1])

                # Calculate average scores across folds
                avg_accuracy = np.mean(fold_accuracies)
                avg_loss = np.mean(fold_losses)
                avg_bce = np.mean(fold_bce)
                avg_mse = np.mean(fold_mse)
                avg_epochs = np.mean(fold_epochs)
                avg_train_acc = np.mean(fold_train_acc)

                # Store results
                results['hidden_neurons'].append(hidden_neuron)
                results['hidden_neurons_2nd_layer'].append(hidden_neuron_2nd_layer)
                results['hidden_neurons_3rd_layer'].append(hidden_neuron_3rd_layer)
                results['val_accuracy'].append(avg_accuracy)
                results['val_loss'].append(avg_loss)
                results['val_binary_crossentropy'].append(avg_bce)
                results['val_mse'].append(avg_mse)
                results['epochs_trained'].append(avg_epochs)
                results['all_histories'].append(all_fold_histories)
                results['train_accuracy'].append(avg_train_acc)

                gap = []

                for i, hidden_neuron in enumerate(results['hidden_neurons']):
                    gap.append(results['train_accuracy'][i] - results['val_accuracy'][i])

                results['generalization_gap'].append(gap)

                print(f"Results - Accuracy: {avg_accuracy:.4f}, Loss: {avg_loss:.4f}, BCE: {avg_bce:.4f}, MSE: {avg_mse:.4f}, Avg Epochs: {avg_epochs:.1f}, Avg train accuracy: {avg_train_acc:.4f}")

    return results

# --- Function to save results to CSV files ---
def save_results_to_csv(results, output_dir):
    # Create DataFrame for summary results (averaged across folds)
    summary_df = pd.DataFrame({
        'hidden_neurons': results['hidden_neurons'],
        'hidden_neurons_2nd_layer': results['hidden_neurons_2nd_layer'],
        'hidden_neurons_3rd_layer': results['hidden_neurons_3rd_layer'],
        'val_accuracy': results['val_accuracy'],
        'val_loss': results['val_loss'],
        'val_binary_crossentropy': results['val_binary_crossentropy'],
        'val_mse': results['val_mse'],
        'epochs_trained': results['epochs_trained'],
        'train_accuracy': results['train_accuracy'],
        'generalization_gap': results['generalization_gap']
    })

    # Save summary results
    summary_filename = os.path.join(output_dir, "third_layer_neurons_count_tuning.csv")
    summary_df.to_csv(summary_filename, index=False)
    print(f"Summary results saved to {summary_filename}")

    return summary_filename

# --- Function to plot convergence graphs for each hyperparameter combination ---
def plot_convergence_graphs(results, output_dir):
    metrics = ['loss', 'val_loss', 'accuracy', 'val_accuracy']
    metric_titles = ['Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy']
    filenames = []

    for metric, title in zip(metrics, metric_titles):
        plt.figure(figsize=(12, 8))

        for i, neurons_histories in enumerate(results['all_histories']):
            hidden_neuron = results['hidden_neurons'][i]
            hidden_neuron_2nd_layer = results['hidden_neurons_2nd_layer'][i]
            hidden_neuron_3rd_layer = results['hidden_neurons_3rd_layer'][i]

            # Average the metric across all folds
            max_epochs = max([len(fold_hist[metric]) for fold_hist in neurons_histories])
            avg_metric = np.zeros(max_epochs)
            counts = np.zeros(max_epochs)

            for fold_hist in neurons_histories:
                epochs_in_fold = len(fold_hist[metric])
                avg_metric[:epochs_in_fold] += fold_hist[metric]
                counts[:epochs_in_fold] += 1

            # Avoid division by zero
            counts[counts == 0] = 1
            avg_metric = avg_metric / counts

            # Plot the average metric
            plt.plot(avg_metric, label=f'{hidden_neuron} hidden neurons in first hidden layer, {hidden_neuron_2nd_layer} hidden neurons in second hidden layer and {hidden_neuron_3rd_layer} hidden neurons in third hidden layer')

        plt.title(f'{title} vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(title)
        plt.legend()
        plt.grid(True)

        # Save the figure
        filename = os.path.join(output_dir, f"convergence_{metric}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        filenames.append(filename)
        print(f"Saved convergence graph to {filename}")

    return filenames

# --- Complete workflow function ---
def run_hyperparameter_analysis(X, y, hidden_neurons, hidden_neurons_2nd_layer, hidden_neurons_3rd_layer):
    """Complete workflow for hyperparameter analysis with CSV and PNG exports"""
    # Create output directory
    output_dir = create_output_directory()

    # Evaluate hyperparameters
    results = evaluate_hyperparameters(X, y, hidden_neurons, hidden_neurons_2nd_layer, hidden_neurons_3rd_layer)

    # Save results to CSV
    summary_file = save_results_to_csv(results, output_dir)

    # Generate and save plots
    convergence_plots = plot_convergence_graphs(results, output_dir)

    # Find the best hyperparameter combination
    best_idx = np.argmax(results['val_accuracy'])

    print("\n=== Hyperparameter Analysis Complete ===")
    print(f"Results saved to directory: {output_dir}")
    print(f"\nBest hyperparameter combination:")
    print(f"Hidden neurons count: {results['hidden_neurons'][best_idx]}")
    print(f"Hidden neurons count in second layer: {results['hidden_neurons_2nd_layer'][best_idx]}")
    print(f"Hidden neurons count in third layer: {results['hidden_neurons_3rd_layer'][best_idx]}")
    print(f"Validation Accuracy: {results['val_accuracy'][best_idx]:.4f}")
    print(f"Validation Loss: {results['val_loss'][best_idx]:.4f}")

    return results, output_dir

results, output_dir = run_hyperparameter_analysis(X, y, hidden_neurons, hidden_neurons_2nd_layer, hidden_neurons_3rd_layer)

print("\n✅ Model Training and Evaluation Complete!")
