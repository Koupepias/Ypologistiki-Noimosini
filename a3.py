import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
from tensorflow.keras import backend as K
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
    output_dir = f"hyperparameter_results_{timestamp}"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    return output_dir

# Define hyperparameter combinations to test
lr_momentum_combos = [
    [0.001, 0.2],
    [0.001, 0.6],
    [0.05, 0.6],
    [0.1, 0.6]
]

# --- Define a function to build the Neural Network with SGD optimizer ---
def build_model(learning_rate, momentum):
    model = keras.Sequential([
        layers.Dense(76, activation="tanh"),  # Hidden layer
        layers.Dense(1, activation='sigmoid')  # Output layer 
    ])
    
    # Use SGD optimizer with momentum instead of Adam
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    
    model.compile(optimizer=optimizer,
                  loss="mse",
                  metrics=['accuracy', 'binary_crossentropy', 'mse'])
    return model

# --- Early Stopping Callback ---
early_stopping_val_loss = EarlyStopping(
    monitor='val_loss',
    patience=10,              
    min_delta=0.0005,         
    restore_best_weights=True,
    verbose=1,
    mode='min'
)

# --- Function to evaluate hyperparameter combinations ---
def evaluate_hyperparameters(X, y, lr_momentum_combos):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize dictionaries to store results
    results = {
        'combo': [],
        'lr': [],
        'momentum': [],
        'val_accuracy': [],
        'val_loss': [],
        'val_binary_crossentropy': [],
        'val_mse': [],
        'epochs_trained': [],
        'all_histories': []  # Store all training histories
    }
    
    # Also store fold-level metrics for more detailed analysis
    #fold_results = []
    
    # Evaluate each hyperparameter combination
    for combo in lr_momentum_combos:
        learning_rate, momentum = combo
        print(f"\nEvaluating learning_rate={learning_rate}, momentum={momentum}")
        
        fold_accuracies = []
        fold_losses = []
        fold_bce = []
        fold_mse = []
        fold_epochs = []
        all_fold_histories = []
        val_scores = []
        
        # Perform k-fold cross-validation
        for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
            print(f"  Fold {fold + 1}/5")
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Build model with current hyperparameters
            model = build_model(learning_rate, momentum)
            
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
            
            # Add to fold-level results
            # fold_results.append({
            #     'lr': learning_rate,
            #     'momentum': momentum,
            #     'fold': fold + 1,
            #     'val_accuracy': val_accuracy,
            #     'val_loss': val_loss,
            #     'val_binary_crossentropy': val_bce,
            #     'val_mse': val_mse,
            #     'epochs': epochs
            # })
        
        # Calculate average scores across folds
        avg_accuracy = np.mean(fold_accuracies)
        avg_loss = np.mean(fold_losses)
        avg_bce = np.mean(fold_bce)
        avg_mse = np.mean(fold_mse)
        avg_epochs = np.mean(fold_epochs)
        
        # Store results
        combo_str = f"LR={learning_rate}, M={momentum}"
        results['combo'].append(combo_str)
        results['lr'].append(learning_rate)
        results['momentum'].append(momentum)
        results['val_accuracy'].append(avg_accuracy)
        results['val_loss'].append(avg_loss)
        results['val_binary_crossentropy'].append(avg_bce)
        results['val_mse'].append(avg_mse)
        results['epochs_trained'].append(avg_epochs)
        results['all_histories'].append(all_fold_histories)
        
        print(f"  Results - Accuracy: {avg_accuracy:.4f}, Loss: {avg_loss:.4f}, BCE: {avg_bce:.4f}, MSE: {avg_mse:.4f}, Avg Epochs: {avg_epochs:.1f}")
    
    return results #, fold_results

# --- Function to save results to CSV files ---
def save_results_to_csv(results, output_dir):
    # Create DataFrame for summary results (averaged across folds)
    summary_df = pd.DataFrame({
        'learning_rate': results['lr'],
        'momentum': results['momentum'],
        'val_accuracy': results['val_accuracy'],
        'val_loss': results['val_loss'],
        'val_binary_crossentropy': results['val_binary_crossentropy'],
        'val_mse': results['val_mse'],
        'epochs_trained': results['epochs_trained']
    })
    
    # Save summary results
    summary_filename = os.path.join(output_dir, "hyperparameter_summary.csv")
    summary_df.to_csv(summary_filename, index=False)
    print(f"Summary results saved to {summary_filename}")
    
    # Create DataFrame for detailed fold results
    #fold_df = pd.DataFrame(fold_results)
    
    # Save detailed fold results
    # fold_filename = os.path.join(output_dir, "hyperparameter_folds.csv")
    # fold_df.to_csv(fold_filename, index=False)
    # print(f"Detailed fold results saved to {fold_filename}")
    
    # # Create convergence data for each hyperparameter combination
    # convergence_filenames = []
    # for i, combo in enumerate(results['combo']):
    #     # Extract learning rate and momentum for this combo
    #     lr = results['lr'][i]
    #     momentum = results['momentum'][i]
        
    #     # Process histories for this combo
    #     all_histories = results['all_histories'][i]
    #     max_epochs = max([len(fold_hist['loss']) for fold_hist in all_histories])
        
    #     # Create DataFrame to store epoch-by-epoch metrics for all folds
    #     history_data = []
        
    #     for fold, history in enumerate(all_histories):
    #         for epoch in range(len(history['loss'])):
    #             history_data.append({
    #                 'learning_rate': lr,
    #                 'momentum': momentum,
    #                 'fold': fold + 1,
    #                 'epoch': epoch + 1,
    #                 'loss': history['loss'][epoch],
    #                 'val_loss': history['val_loss'][epoch],
    #                 'accuracy': history['accuracy'][epoch],
    #                 'val_accuracy': history['val_accuracy'][epoch],
    #                 'binary_crossentropy': history['binary_crossentropy'][epoch],
    #                 'val_binary_crossentropy': history['val_binary_crossentropy'][epoch],
    #                 'mse': history['mse'][epoch],
    #                 'val_mse': history['val_mse'][epoch]
    #             })
        
    #     # Create DataFrame and save to CSV
    #     history_df = pd.DataFrame(history_data)
    #     history_filename = os.path.join(output_dir, f"convergence_lr{lr}_momentum{momentum}.csv")
    #     history_df.to_csv(history_filename, index=False)
    #     print(f"Convergence data for LR={lr}, M={momentum} saved to {history_filename}")
    #     convergence_filenames.append(history_filename)
    
    return summary_filename #, fold_filename, convergence_filenames

# --- Function to plot convergence graphs for each hyperparameter combination ---
def plot_convergence_graphs(results, output_dir):
    metrics = ['loss', 'val_loss', 'accuracy', 'val_accuracy']
    metric_titles = ['Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy']
    filenames = []
    
    for metric, title in zip(metrics, metric_titles):
        plt.figure(figsize=(12, 8))
        
        for i, combo_histories in enumerate(results['all_histories']):
            lr = results['lr'][i]
            momentum = results['momentum'][i]
            combo_str = f"LR={lr}, M={momentum}"
            
            # Average the metric across all folds
            max_epochs = max([len(fold_hist[metric]) for fold_hist in combo_histories])
            avg_metric = np.zeros(max_epochs)
            counts = np.zeros(max_epochs)
            
            for fold_hist in combo_histories:
                epochs_in_fold = len(fold_hist[metric])
                avg_metric[:epochs_in_fold] += fold_hist[metric]
                counts[:epochs_in_fold] += 1
            
            # Avoid division by zero
            counts[counts == 0] = 1
            avg_metric = avg_metric / counts
            
            # Plot the average metric
            plt.plot(avg_metric, label=combo_str)
        
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

# --- Function to plot loss and accuracy against learning rate in subplots --- 
def plot_loss_vs_lr(results, output_dir):
    filenames = []
    
    # Create groups by momentum value
    momentum_values = list(set(results['momentum']))
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot loss vs learning rate (left subplot)
    for momentum in momentum_values:
        # Find indices for this momentum value
        indices = [i for i, m in enumerate(results['momentum']) if m == momentum]
        
        # Extract learning rates and corresponding losses
        lr_values = [results['lr'][i] for i in indices]
        loss_values = [results['val_loss'][i] for i in indices]
        
        # Sort by learning rate
        sorted_indices = np.argsort(lr_values)
        lr_values = [lr_values[i] for i in sorted_indices]
        loss_values = [loss_values[i] for i in sorted_indices]
        
        # Plot on first subplot
        ax1.plot(lr_values, loss_values, 'o-', label=f'Momentum = {momentum}')
    
    ax1.set_title('Validation Loss vs. Learning Rate')
    ax1.set_xlabel('Learning Rate (log scale)')
    ax1.set_ylabel('Validation Loss (MSE)')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy vs learning rate (right subplot)
    for momentum in momentum_values:
        # Find indices for this momentum value
        indices = [i for i, m in enumerate(results['momentum']) if m == momentum]
        
        # Extract learning rates and corresponding accuracy
        lr_values = [results['lr'][i] for i in indices]
        acc_values = [results['val_accuracy'][i] for i in indices]
        
        # Sort by learning rate
        sorted_indices = np.argsort(lr_values)
        lr_values = [lr_values[i] for i in sorted_indices]
        acc_values = [acc_values[i] for i in sorted_indices]
        
        # Plot on second subplot
        ax2.plot(lr_values, acc_values, 'o-', label=f'Momentum = {momentum}')
    
    ax2.set_title('Validation Accuracy vs. Learning Rate')
    ax2.set_xlabel('Learning Rate (log scale)')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True)
    
    # Add a common title for the entire figure
    plt.suptitle('Model Performance vs Learning Rate by Momentum Value', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle
    
    # Save the figure
    filename = os.path.join(output_dir, "performance_vs_lr.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    filenames.append(filename)
    print(f"Saved performance vs learning rate graph to {filename}")
    
    return filenames

# --- Function to plot heatmap of hyperparameter performance ---
def plot_heatmap(results, output_dir):
    # Check if we have enough data points for a meaningful heatmap
    unique_lr = sorted(list(set(results['lr'])))
    unique_momentum = sorted(list(set(results['momentum'])))
    
    if len(unique_lr) > 1 and len(unique_momentum) > 1:
        # Create matrices for the heatmap
        accuracy_matrix = np.zeros((len(unique_momentum), len(unique_lr)))
        loss_matrix = np.zeros((len(unique_momentum), len(unique_lr)))
        
        # Fill in the matrices
        for i, mom in enumerate(unique_momentum):
            for j, lr in enumerate(unique_lr):
                # Find the index in results where lr and momentum match
                indices = [idx for idx, (l, m) in enumerate(zip(results['lr'], results['momentum'])) 
                          if l == lr and m == mom]
                
                if indices:
                    accuracy_matrix[i, j] = results['val_accuracy'][indices[0]]
                    loss_matrix[i, j] = results['val_loss'][indices[0]]
                else:
                    accuracy_matrix[i, j] = np.nan
                    loss_matrix[i, j] = np.nan
        
        # Plot accuracy heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(accuracy_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Validation Accuracy')
        
        # Set the ticks and labels
        plt.xticks(range(len(unique_lr)), [str(lr) for lr in unique_lr])
        plt.yticks(range(len(unique_momentum)), [str(m) for m in unique_momentum])
        plt.xlabel('Learning Rate')
        plt.ylabel('Momentum')
        
        # Add text annotations
        for i in range(len(unique_momentum)):
            for j in range(len(unique_lr)):
                if not np.isnan(accuracy_matrix[i, j]):
                    plt.text(j, i, f"{accuracy_matrix[i, j]:.3f}", 
                            ha="center", va="center", color="white" if accuracy_matrix[i, j] < 0.7 else "black")
        
        plt.title('Validation Accuracy Heatmap')
        
        # Save the figure
        filename = os.path.join(output_dir, "accuracy_heatmap.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved accuracy heatmap to {filename}")
        
        # Plot loss heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(loss_matrix, cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label='Validation Loss (MSE)')
        
        # Set the ticks and labels
        plt.xticks(range(len(unique_lr)), [str(lr) for lr in unique_lr])
        plt.yticks(range(len(unique_momentum)), [str(m) for m in unique_momentum])
        plt.xlabel('Learning Rate')
        plt.ylabel('Momentum')
        
        # Add text annotations
        for i in range(len(unique_momentum)):
            for j in range(len(unique_lr)):
                if not np.isnan(loss_matrix[i, j]):
                    plt.text(j, i, f"{loss_matrix[i, j]:.3f}", 
                            ha="center", va="center", color="white" if loss_matrix[i, j] > 0.25 else "black")
        
        plt.title('Validation Loss Heatmap')
        
        # Save the figure
        filename = os.path.join(output_dir, "loss_heatmap.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved loss heatmap to {filename}")
        
        return [os.path.join(output_dir, "accuracy_heatmap.png"), 
                os.path.join(output_dir, "loss_heatmap.png")]
    else:
        print("Not enough unique values for learning rate and momentum to create meaningful heatmaps")
        return []

# --- Function to plot comparison of all hyperparameters ---
def plot_hyperparameter_comparison(results, output_dir):
    # Create bar plots for key metrics
    metrics = ['val_accuracy', 'val_loss', 'val_binary_crossentropy', 'epochs_trained']
    titles = ['Validation Accuracy', 'Validation Loss (MSE)', 'Binary Crossentropy', 'Avg Epochs']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        axes[i].bar(results['combo'], results[metric])
        axes[i].set_title(title)
        axes[i].set_xticklabels(results['combo'], rotation=45)
        if metric == 'val_accuracy':
            axes[i].set_ylim(0, 1)
        axes[i].grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the figure
    filename = os.path.join(output_dir, "hyperparameter_comparison.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved hyperparameter comparison to {filename}")
    
    return filename

# --- Complete workflow function ---
def run_hyperparameter_analysis(X, y, lr_momentum_combos):
    """Complete workflow for hyperparameter analysis with CSV and PNG exports"""
    # Create output directory
    output_dir = create_output_directory()
    
    # Evaluate hyperparameters
    results = evaluate_hyperparameters(X, y, lr_momentum_combos)
    
    # Save results to CSV
    summary_file = save_results_to_csv(results, output_dir)
    
    # Generate and save plots
    convergence_plots = plot_convergence_graphs(results, output_dir)
    lr_plots = plot_loss_vs_lr(results, output_dir)
    heatmap_plots = plot_heatmap(results, output_dir)
    comparison_plot = plot_hyperparameter_comparison(results, output_dir)
    
    # Find the best hyperparameter combination
    best_idx = np.argmax(results['val_accuracy'])
    
    print("\n=== Hyperparameter Analysis Complete ===")
    print(f"Results saved to directory: {output_dir}")
    print(f"\nBest hyperparameter combination:")
    print(f"Learning Rate: {results['lr'][best_idx]}")
    print(f"Momentum: {results['momentum'][best_idx]}")
    print(f"Validation Accuracy: {results['val_accuracy'][best_idx]:.4f}")
    print(f"Validation Loss: {results['val_loss'][best_idx]:.4f}")
    
    return results, output_dir

results, output_dir = run_hyperparameter_analysis(X, y, lr_momentum_combos)
print("\n✅ Model Training and Evaluation Complete!")
