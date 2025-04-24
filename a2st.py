import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, Callback
from datetime import datetime
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
    output_dir = f"a2st_{timestamp}"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    return output_dir

# --- Define a function to build the Neural Network with SGD optimizer ---
def build_model(activation_function, loss_function, hidden_neurons_count ):
    model = keras.Sequential([
        layers.Dense(hidden_neurons_count, activation=activation_function),  # Hidden layer
        layers.Dense(1, activation='sigmoid')  # Output layer 
    ])
        
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.001),
                  loss=loss_function,
                  metrics=['accuracy', 'binary_crossentropy', 'mse'])
    return model

I=X.shape[1]  # Number of features
print(f"Number of features in X: {I}")

# 1. Standard Early Stopping (based on validation loss)
early_stopping_val_loss = EarlyStopping(
    monitor='val_loss',
    patience=10,              # Stop after 5 epochs of no improvement
    min_delta=0.001,         # Minimum change to qualify as improvement
    restore_best_weights=True,
    verbose=1,
    mode='min'
)

# 2. Early Stopping based on validation accuracy
early_stopping_val_acc = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    min_delta=0.001,         # 0.1% improvement threshold
    restore_best_weights=True,
    verbose=1,
    mode='max'               # We want to maximize accuracy
)

# 3. Custom Callback for convergence threshold
class CombinedMetricStopping(Callback):
    def __init__(self, loss_patience=8, acc_patience=8, min_loss_delta=0.001, min_acc_delta=0.005):
        super().__init__()
        self.loss_patience = loss_patience
        self.acc_patience = acc_patience
        self.min_loss_delta = min_loss_delta
        self.min_acc_delta = min_acc_delta
        self.best_loss = float('inf')
        self.best_acc = -float('inf')
        self.loss_wait = 0
        self.acc_wait = 0
    
    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        current_acc = logs.get('val_accuracy')
        
        if current_loss < (self.best_loss - self.min_loss_delta):
            self.best_loss = current_loss
            self.loss_wait = 0
        else:
            self.loss_wait += 1
            
        if current_acc > (self.best_acc + self.min_acc_delta):
            self.best_acc = current_acc
            self.acc_wait = 0
        else:
            self.acc_wait += 1
            
        if self.loss_wait >= self.loss_patience and self.acc_wait >= self.acc_patience:
            self.model.stop_training = True
            print(f'\nStopping - Loss hasn\'t improved for {self.loss_wait} epochs and Accuracy for {self.acc_wait} epochs')

class LearningCurveStabilization(Callback):
    def __init__(self, window_size=5, threshold=0.01):
        super().__init__()
        self.window_size = window_size
        self.threshold = threshold
        self.train_losses = []
        self.val_losses = []
        self.best_weights = None
        self.best_epoch = 0
        
    def on_epoch_end(self, epoch, logs=None):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        
        # Need enough history to make a decision
        if len(self.train_losses) >= self.window_size * 2:
            # Calculate average slope of recent training and validation losses
            train_window1 = self.train_losses[-2*self.window_size:-self.window_size]
            train_window2 = self.train_losses[-self.window_size:]
            val_window1 = self.val_losses[-2*self.window_size:-self.window_size]
            val_window2 = self.val_losses[-self.window_size:]
            
            train_slope = (np.mean(train_window2) - np.mean(train_window1)) / self.window_size
            val_slope = (np.mean(val_window2) - np.mean(val_window1)) / self.window_size
            
            # Check if both slopes are small (stable)
            if abs(train_slope) < self.threshold and abs(val_slope) < self.threshold:
                # Save best weights based on validation loss
                current_val_loss = self.val_losses[-1]
                min_val_loss = min(self.val_losses)
                best_epoch = self.val_losses.index(min_val_loss)
                
                print(f"\nEpoch {epoch+1}: Learning curves have stabilized. Stopping training.")
                print(f"Best validation loss was at epoch {best_epoch+1}")
                
                self.model.stop_training = True

combo = [["elu", "bce", 25],
        ["elu", "bce", 76],
        ["relu", "bce", 76],
        [layers.LeakyReLU(negative_slope=0.01), "bce", 76],
        [layers.LeakyReLU(negative_slope=0.01), "bce", 38],
        ["swish", "bce", 76]
]

callbacks= [early_stopping_val_loss, 
            early_stopping_val_acc, 
            CombinedMetricStopping(loss_patience=4, acc_patience=4),
            LearningCurveStabilization(window_size=5, threshold=0.01)

]

# --- Function to evaluate hyperparameter combinations ---
def evaluate_hyperparameters(X, y, combo, callbacks):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize dictionaries to store results
    results = {
        'activation_function': [],
        'loss_function': [],
        'hidden_neurons_count': [],
        'stopping_criterion': [],
        'val_accuracy': [],
        'val_loss': [],
        'val_binary_crossentropy': [],
        'val_mse': [],
        'epochs_trained': [],
        'train_accuracy': [],
        'all_histories': []  # Store all training histories
    }
    
    # Evaluate each hyperparameter combination
    for params in combo:
        activation_function, loss_function, hidden_neurons_count = params
        for callback in callbacks:
            print(f"\nEvaluating {activation_function} + {loss_function} + {hidden_neurons_count} neurons with {callback}")

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
                model = build_model(activation_function, loss_function, hidden_neurons_count)
                
                # Train the model
                history = model.fit(
                    X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    verbose=0,
                    validation_data=(X_test, y_test),
                    callbacks=[callback]  
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
            results['activation_function'].append(activation_function)
            results['loss_function'].append(loss_function)
            results['hidden_neurons_count'].append(hidden_neurons_count)
            results['stopping_criterion'].append(callback.__class__.__name__)
            results['val_accuracy'].append(avg_accuracy)
            results['val_loss'].append(avg_loss)
            results['val_binary_crossentropy'].append(avg_bce)
            results['val_mse'].append(avg_mse)
            results['epochs_trained'].append(avg_epochs)
            results['all_histories'].append(all_fold_histories)
            results['train_accuracy'].append(avg_train_acc)

            print(f"Model: {activation_function} & {loss_function} with {hidden_neurons_count}")
            print(f"  Results - Accuracy: {avg_accuracy:.4f}, Loss: {avg_loss:.4f}, BCE: {avg_bce:.4f}, MSE: {avg_mse:.4f}, Avg Epochs: {avg_epochs:.1f}, Avg train accuracy: {avg_train_acc:.4f}")

    return results

# --- Function to save results to CSV files ---
def save_results_to_csv(results, output_dir):
    # Create DataFrame for summary results (averaged across folds)
    summary_df = pd.DataFrame({
        'activation_function': results['activation_function'],
        'loss_function': results['loss_function'],
        'hidden_neurons_count': results['hidden_neurons_count'],
        'stopping_criterion': results['stopping_criterion'],
        'val_accuracy': results['val_accuracy'],
        'val_loss': results['val_loss'],
        'val_binary_crossentropy': results['val_binary_crossentropy'],
        'val_mse': results['val_mse'],
        'epochs_trained': results['epochs_trained'],
        'train_accuracy': results['train_accuracy'],
    })
    
    # Save summary results
    summary_filename = os.path.join(output_dir, "act-loss-neurons-tuning-termination_criterion.csv")
    summary_df.to_csv(summary_filename, index=False)
    print(f"Summary results saved to {summary_filename}")
    print("Best 5 combinations based on accuracy:")
    print(summary_df.nlargest(5, 'val_accuracy'))

    return summary_filename

# --- Function to plot convergence graphs for each hyperparameter combination ---
def plot_convergence_graphs(results, output_dir):
    metrics = ['loss', 'val_loss', 'accuracy', 'val_accuracy']
    metric_labels = ['Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy']
    metric_colors = ['blue', 'red', 'green', 'orange']
    filenames = []
    
    # For each hyperparameter combination
    for i, histories in enumerate(results['all_histories']):
        act = results['activation_function'][i]
        loss = results['loss_function'][i]
        neurons = results['hidden_neurons_count'][i]
        
        # Create a new figure for this combination
        plt.figure(figsize=(12, 8))
        
        # Plot each metric
        for metric, label, color in zip(metrics, metric_labels, metric_colors):
            # Average the metric across all folds
            max_epochs = max([len(fold_hist[metric]) for fold_hist in histories])
            avg_metric = np.zeros(max_epochs)
            counts = np.zeros(max_epochs)
            
            for fold_hist in histories:
                epochs_in_fold = len(fold_hist[metric])
                avg_metric[:epochs_in_fold] += fold_hist[metric]
                counts[:epochs_in_fold] += 1
            
            # Avoid division by zero
            counts[counts == 0] = 1
            avg_metric = avg_metric / counts

            # Plot this metric
            plt.plot(avg_metric, label=label, color=color)

            # Add marker for best epoch based on validation loss
            if metric == 'val_loss':
                min_val_loss_epoch = np.argmin(avg_metric)
                plt.axvline(x=min_val_loss_epoch, color='purple', linestyle='--', 
                          label=f'Best Epoch: {min_val_loss_epoch}')
        
        # Set plot title and labels
        plt.title(f'Training Metrics\nActivation: {act}, Loss: {loss}, Neurons: {neurons}')
        plt.xlabel('Epochs')
        plt.ylabel('Training Loss, Validation Loss, Training Accuracy, Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Save the figure
        act_name = act if isinstance(act, str) else act.__class__.__name__
        filename = os.path.join(output_dir, f'{act_name}-{loss}-{neurons}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        filenames.append(filename)
        print(f"Saved combined metrics plot to {filename}")
    
    return filenames

# --- Complete workflow function ---
def run_hyperparameter_analysis(X, y, combo, callbacks):
    """Complete workflow for hyperparameter analysis with CSV and PNG exports"""
    # Create output directory
    output_dir = create_output_directory()
    
    # Evaluate hyperparameters
    results = evaluate_hyperparameters(X, y,combo, callbacks)
    
    # Save results to CSV
    summary_file = save_results_to_csv(results, output_dir)    
    
    # Generate and save plots
    convergence_plots = plot_convergence_graphs(results, output_dir)

    # Find the best hyperparameter combination
    best_idx = np.argmax(results['val_accuracy'])
    
    print("\n=== Hyperparameter Analysis Complete ===")
    print(f"Results saved to directory: {output_dir}")
    print(f"\nBest hyperparameter combination:")
    print(f"Activation Function: {results['activation_function'][best_idx]}")
    print(f"Loss Function: {results['loss_function'][best_idx]}")
    print(f"Hidden Neurons: {results['hidden_neurons_count'][best_idx]}")
    print(f"Stopping Criterion: {results['stopping_criterion'][best_idx]}")
    print(f"Validation Accuracy: {results['val_accuracy'][best_idx]:.4f}")
    print(f"Validation Loss: {results['val_loss'][best_idx]:.4f}")
    
    return results, output_dir

results, output_dir = run_hyperparameter_analysis(X, y, combo, callbacks)

print("\n✅ Model Training and Evaluation Complete!")
