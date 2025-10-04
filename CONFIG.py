##### CONFIG.py (XConv)
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
filtered_num = 30 # filtering num of SMILES
random_pick_num = 10000 # num_pick
data_extraction_folder = fr"C:\Users\wisdo\polyOne_Data_Set\data_extraction"
chemical_feature_extraction_folder = fr"C:\Users\wisdo\polyOne_Data_Set\chemical_feature_extraction"
graph_feature_extraction_folder = fr"C:\Users\wisdo\polyOne_Data_Set\graph_feature_extraction_folder"
model_save_folder = fr"C:\Users\wisdo\polyOne_Data_Set\models"
plot_save_folder = fr"C:\Users\wisdo\polyOne_Data_Set\plot"

ECFP_radius = 2
ECFP_nBits = 1024

add_hydrogen= True
use_pos = True

batch_size = 256
learning_rate = 3e-4
epochs = 100

# LR scheduler setting
StepLR_step_size = 10
StepLR_gamma= 0.2

ROnPlateauLR_mode = 'min'
ROnPlateauLR_factor = 0.2
ROnPlateauLR_patience = 5


