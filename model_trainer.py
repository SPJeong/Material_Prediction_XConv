##### model_trainer.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Training function with AMP
def train(model, data_loader, loss_fn, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for data_graph, data_descriptor_ECFP in data_loader['train']:
        data_graph = data_graph.to(device)  # to cuda
        data_descriptor_ECFP = data_descriptor_ECFP.to(device)  # to cuda
        optimizer.zero_grad()

        # autocast context for forward calculation
        with autocast():
            outputs = model(data_graph, data_descriptor_ECFP)
            outputs = outputs[:, 0]
            loss = loss_fn(outputs, data_graph.y)

        # scaler for backward and param update
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = data_graph.y.size(0)  # NOT use of len(data_graph.y.size(0))
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_train_loss = total_loss / total_samples
    return avg_train_loss


# Validation function with AMP (autocast only)
def validate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for data_graph, data_descriptor_ECFP in data_loader['val']:
            data_graph = data_graph.to(device)
            data_descriptor_ECFP = data_descriptor_ECFP.to(device)

            with autocast():
                outputs = model(data_graph, data_descriptor_ECFP)
                outputs = outputs[:, 0]
                loss = loss_fn(outputs, data_graph.y)

            batch_size = data_graph.y.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_val_loss = total_loss / total_samples
    return avg_val_loss


# Test function using MAE with AMP (autocast only)
def test(model, data_loader, device, plot_save_folder, model_name, target_name):
    model.eval()
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for data_graph, data_descriptor_ECFP in data_loader['test']:
            data_graph = data_graph.to(device)
            data_descriptor_ECFP = data_descriptor_ECFP.to(device)
            targets = data_graph.y

            with autocast():
                outputs = model(data_graph, data_descriptor_ECFP)
                outputs = outputs[:, 0]

            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all outputs and targets into a single tensor, then convert to numpy arrays
    all_outputs = torch.cat(all_outputs).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # Compute metrics using scikit-learn
    mse = mean_squared_error(all_targets, all_outputs)
    mae = mean_absolute_error(all_targets, all_outputs)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_outputs)

    # Add Plotting and Saving
    # Create the scatter plot
    os.makedirs(plot_save_folder, exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.scatter(all_targets, all_outputs, alpha=0.5, color='blue', label='Predictions')

    # Add a line of perfect prediction line (y=x) for reference
    min_val = min(np.min(all_targets), np.min(all_outputs))  # numpy min
    max_val = max(np.max(all_targets), np.max(all_outputs))  # numpy max

    # Add padding space
    padding = (max_val - min_val) * 0.05
    plot_min = min_val - padding
    plot_max = max_val + padding

    plt.plot([plot_min, plot_max], [plot_min, plot_max], color='red', linestyle='--', lw=2, label='Perfect Prediction')

    # Add labels, title, and a legend
    plt.xlim(plot_min, plot_max)
    plt.ylim(plot_min, plot_max)
    plt.xlabel(f'Actual {target_name}', fontsize=12)
    plt.ylabel(f'Predicted {target_name}', fontsize=12)
    plt.title(f'{model_name}: Actual vs. Predicted ({target_name}) \n(MAE: {mae:.4f}, R2: {r2:.4f})', fontsize=14)
    plt.legend()
    plt.grid(True)

    # Save the plot as an image file
    plot_save_path = os.path.join(plot_save_folder, f'{model_name}_MAE_plot_{target_name}.png')
    plt.savefig(plot_save_path, bbox_inches='tight')
    plt.close()
    print(f'{model_name} MAE plot saved to {plot_save_path}')

    return {'test_mse': mse, 'test_mae': mae, 'test_rmse': rmse, 'test_r2': r2}
