##### my_utils.py (XConv)

import pandas as pd
import torch
import torch_geometric
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class myGNN_Dataset(torch_geometric.data.Dataset):
    def __init__(self, data_df):
        super().__init__()

        self.data_df = data_df
        start_column_index = self.data_df.columns.get_loc('0')
        end_column_index = self.data_df.columns.get_loc('CalcNumBridgeheadAtoms')
        self.ECFP_descriptor_df = self.data_df.iloc[:, start_column_index:end_column_index + 1].copy()

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        # get row from dataframe
        row = self.data_df.iloc[idx]
        # get PyG Data object
        graph_data = row['graph']

        # confirm float32 for graph_data except for edge index data
        for key, value in graph_data:
            if torch.is_tensor(value) and value.dtype in [torch.float64, torch.float16]:
                graph_data[key] = value.float()

        # get descriptor_ECFP and convert into tensor
        # df -> numpy (using .values) -> tensor (using .from_numpy)
        ECFP_descriptor_df_data = torch.from_numpy(self.ECFP_descriptor_df.iloc[idx].values).float()  # float32

        return graph_data, ECFP_descriptor_df_data


def prepare_graph_data_loaders(total_df, target_name, batch_size, random_state=777):
    """
    Args:
        total_df: dataframe containing (graph + ECFP/Descriptor + all_targets)
        target_name: target_name
        batch_size: batch_size
        random_state: default 777
    """
    loaded_graph_y = total_df[['smiles', 'graph', str(target_name)]].copy()

    # add Data.y
    print(f"Starting to add target value ({target_name}) to graph data.y...")
    for i in tqdm(range(len(loaded_graph_y['graph'])), desc=f"Adding {target_name} to Data.y"):
        target_value = loaded_graph_y[target_name].iloc[i].item()
        if pd.notna(target_value):
            loaded_graph_y['graph'].iloc[i].y = torch.tensor([target_value], dtype=torch.float)
        else:
            print(f"\n[WARNING: NaN Value Found] Target '{target_name}' is NaN at index {i}.")
            print(f"Skipping y assignment for SMILES: {smiles}")
            pass

    # ECFP + descriptors for X
    ECFP_descriptor_df_raw = total_df.copy()
    start_column_index = ECFP_descriptor_df_raw.columns.get_loc('0')
    end_column_index = ECFP_descriptor_df_raw.columns.get_loc('CalcNumBridgeheadAtoms')
    ECFP_descriptor_df = ECFP_descriptor_df_raw.iloc[:, start_column_index:end_column_index + 1].copy()

    CONCAT_TOTAL_df_raw = pd.concat([ECFP_descriptor_df, loaded_graph_y, ], axis=1)

    # split
    train, val_test_temp = train_test_split(CONCAT_TOTAL_df_raw, test_size=0.2, random_state=random_state)
    val, test = train_test_split(val_test_temp, test_size=0.5, random_state=random_state)
    print("Data split completed: Train={}, Val={}, Test={}".format(len(train), len(val), len(test)))

    # prepare dataset instance
    train_dataset = myGNN_Dataset(train)
    val_dataset = myGNN_Dataset(val)
    test_dataset = myGNN_Dataset(test)

    # prepare datalaoder
    train_loader = torch_geometric.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch_geometric.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch_geometric.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    data_loader = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    print("DataLoaders created.")

    return data_loader



