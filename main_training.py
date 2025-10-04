###### main_training.py (XConv)
import os
import numpy as np
import pandas as pd
import torch_geometric
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import AllChem

import CONFIG  # custom.py
import chemical_feature_extraction  # custom.py
import graph_3D_feature_extraction  # custom.py

import data_extraction  # custom.py
import model  # custom.py
import my_utils  # custom.py
from model_trainer import train, validate, test  # custom.py

# parameter setting
filtered_num = CONFIG.filtered_num
random_pick_num = CONFIG.random_pick_num
ecfp_radius = CONFIG.ECFP_radius
ecfp_nbits = CONFIG.ECFP_nBits
data_extraction_folder = CONFIG.data_extraction_folder
chemical_feature_extraction_folder = CONFIG.chemical_feature_extraction_folder
graph_feature_extraction_folder = CONFIG.graph_feature_extraction_folder
plot_save_folder = CONFIG.plot_save_folder
model_save_folder = CONFIG.model_save_folder
os.makedirs(chemical_feature_extraction_folder, exist_ok=True)
os.makedirs(graph_feature_extraction_folder, exist_ok=True)
os.makedirs(plot_save_folder, exist_ok=True)
os.makedirs(model_save_folder, exist_ok=True)
model_name = 'XConv'
batch_size = CONFIG.batch_size
learning_rate = CONFIG.learning_rate
epochs = CONFIG.epochs
device = CONFIG.device
ROnPlateauLR_factor = CONFIG.ROnPlateauLR_factor
ROnPlateauLR_patience = CONFIG.ROnPlateauLR_patience
add_hydrogen = CONFIG.add_hydrogen  # True for XConv
use_pos = CONFIG.use_pos  # True for XConv

Y_total_list = ['Cp', 'Tg', 'Tm', 'Td', 'LOI',
                'YM', 'TSy', 'TSb', 'epsb', 'CED',
                'Egc', 'Egb', 'Eib', 'Ei', 'Eea', 'nc', 'ne',
                'permH2', 'permHe', 'permCH4', 'permCO2', 'permN2', 'permO2',
                'Eat', 'rho', 'Xc', 'Xe']

# chemical feature load file
file_folder = chemical_feature_extraction_folder
file_name = f'chemical_feature_extraction_len_{filtered_num}_num_{random_pick_num}_scaled_True_ECFP_True_desc_True.csv'
file_raw_path = os.path.join(file_folder, file_name)

if os.path.exists(file_raw_path):
    print(f"Loading existing file from: {file_raw_path}")
    file_raw = pd.read_csv(file_raw_path)

else:
    print(f"File not found. Generating data and saving to: {file_raw_path}")
    file_raw = chemical_feature_extraction.run_feature_extraction(filtered_num=filtered_num,
                                                                  random_pick_num=random_pick_num,
                                                                  data_extraction_folder=data_extraction_folder,
                                                                  ecfp=True,
                                                                  descriptors=True,
                                                                  scale_descriptors=True,
                                                                  ecfp_radius=ecfp_radius,
                                                                  ecfp_nbits=ecfp_nbits,
                                                                  chemical_feature_extraction_folder=chemical_feature_extraction_folder,
                                                                  inference_mode=False,
                                                                  new_smiles_list=None)

# graph feature load file
file_folder = graph_feature_extraction_folder
file_name = f'graph_feature_len_{filtered_num}_num_{random_pick_num}_scaled_True_ECFP_True_desc_True_H_True_3D_True_graph.h5'
graph_raw_path = os.path.join(file_folder, file_name)

if os.path.exists(graph_raw_path):
    print(f"Loading existing file from: {graph_raw_path}")
    graph_raw = pd.read_hdf(graph_raw_path, key='graph')

else:
    print(f"File not found.  Generating data and saving to: {graph_raw_path}")

    graph_raw = graph_3D_feature_extraction.filter_graphs_from_df(file_raw=file_raw,
                                                                  smiles_column='smiles',
                                                                  add_hydrogen=add_hydrogen,
                                                                  use_pos=use_pos)

    file_folder = graph_feature_extraction_folder
    file_name = f'graph_feature_len_{filtered_num}_num_{random_pick_num}_scaled_True_ECFP_True_desc_True_H_{add_hydrogen}_3D_{use_pos}_graph.h5'
    graph_raw_file_path = os.path.join(file_folder, file_name)
    graph_raw.to_hdf(graph_raw_file_path, key='graph', mode='w')
    print(f"Graph object is saved at '{graph_raw_file_path}'")

total_df = graph_raw.copy()

for i, target_name in tqdm(enumerate(Y_total_list), total=len(Y_total_list)):

    target_name = str(target_name)
    data_loader = my_utils.prepare_graph_data_loaders(total_df, target_name, batch_size, random_state=777)

    # # checking dataloader
    # print("***"*10)
    # print("first_batch_graph, first_batch_descriptor = next(iter(data_loader['train']))")
    # first_batch_graph, first_batch_descriptor = next(iter(data_loader['train']))
    # print(f"first_batch_graph data: {first_batch_graph}")
    # print(f"first_batch_descriptor shape: {first_batch_descriptor.shape}")
    # print("***"*10)

    # XConv_model, loss fn and optimizer define
    my_model = model.XConv_Model()
    my_model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params=my_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scaler = GradScaler()  # GradScaler for float16 calculation

    # ROnPlateau scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=ROnPlateauLR_factor,
                                                           patience=ROnPlateauLR_patience)

    # Main training and validation loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')  # initiate val loss as infinite
    best_model_state = None  # for best model save during training

    for epoch in range(0, epochs):
        # load train, val function
        train_loss = train(my_model, data_loader, loss_fn, optimizer, scaler, device)  # train_loader
        val_loss = validate(my_model, data_loader, loss_fn, device)  # val_loader

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Check if current validation loss is the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = my_model.state_dict()
            print(f"New best model found at epoch {epoch} with Val Loss: {best_val_loss:.4f}. Model state saved.")

        if epoch % 1 == 0:
            print(
                f"Target: {target_name} | Epoch {epoch} | Train Loss (MSE): {train_loss:.4f} | Val Loss (MSE): {val_loss:.4f}")

    # Final evaluation on the test set after training is complete
    final_metrics = test(my_model, data_loader, device, plot_save_folder, model_name, target_name)  # test_loader
    print(f'\nFinal Metrics on Test Set:')
    for metric_name, value in final_metrics.items():
        print(f"{metric_name.upper()}: {value:.4f}")

    # Dictionary to save all parameters and metrics
    results = {'model_type': model_name,
               'target_variable': target_name,
               'model_state_dict': best_model_state,
               'optimizer_state_dict': optimizer.state_dict(),
               'train_losses': train_losses,
               'val_losses': val_losses,
               'final_test_metrics': final_metrics,
               'epochs': epoch, }

    # Save the entire package (best model + metadata)
    model_file_name = f'{model_name}_model_len_{filtered_num}_num_{random_pick_num}_{target_name}.pth'
    model_save_file_name = os.path.join(model_save_folder, model_file_name)
    torch.save(results, model_save_file_name)
    print(f'Best model and training results saved to {model_save_file_name}')



