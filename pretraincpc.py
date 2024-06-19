import torch
from models.cpc import ContrastivePredictiveCoding
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
from pathlib import Path
import json
from datasets import PAMAP2, Opportunity, MMFit, MotionSense, MHEALTH, WISDM
from sklearn.model_selection import GroupKFold
import numpy as np
import sys
import os

# best settings for cpc on capture24
# lr = 0.0005 wd = 0 epochs = 50 patience = 5 batch_size = 256
exp_id = 0

# Global
if 'SLURM_JOB_ID' in os.environ:
    output_dir = '/netscratch/geissler/BeyondConfusion/outputs/pretrain/'
else:
    output_dir = "./pretrain"


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(device)

PATIENCE = 20
NUM_EPOCHS = 400


if len(sys.argv) > 2:
    dataset_string = sys.argv[1]
    model_string = sys.argv[2]

    DATASET_LIST = [dataset_string]
    MODEL_LIST = [model_string]
else:
    DATASET_LIST = ['OPPORTUNITY']
    MODEL_LIST = ['CPC']

BATCH_LIST = [64, 256, 1024]
LR_LIST = [0.1, 0.01, 0.001]
WINDOW = 200
STRIDE = 100


# DATASET LOOP
for DATASET in DATASET_LIST:
    print("\nDATASET: " + DATASET)

    # Select Dataset
    if DATASET=='PAMAP2':
        ds = PAMAP2(users='full', window_size=WINDOW, window_step=STRIDE, frequency=50, columns=None, train_users=[1, 2, 3, 4, 5, 6, 7, 8])
        INPUT_SIZE = 39
        CLASSES = 13
        SPLIT = len(set(ds.user_id_list))
    elif DATASET=='OPPORTUNITY':
        ds = Opportunity(users=[1, 2, 3, 4], window_size=WINDOW,window_step=STRIDE)
        INPUT_SIZE = 114
        CLASSES = 18
        SPLIT = len(set(ds.user_id_list))
    elif DATASET=='MMFIT':
        ds = MMFit(users='full', window_size=WINDOW, window_step=STRIDE, frequency=50, columns=None, train_users=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']) 
        INPUT_SIZE = 24
        CLASSES = 11
        SPLIT = 7
    elif DATASET=='MHEALTH':
        ds = MHEALTH(users='full', window_size=WINDOW, window_step=STRIDE, train_users=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) 
        INPUT_SIZE = 21
        CLASSES = 13
        SPLIT = len(set(ds.user_id_list))
    elif DATASET=='MOTIONSENSE':
        ds = MotionSense(users='full', window_size=WINDOW, window_step=STRIDE, train_users=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]) 
        INPUT_SIZE = 12
        CLASSES = 6
        SPLIT = 8
    elif DATASET=='WISDM':
        ds = WISDM(users='full', window_size=WINDOW, window_step=STRIDE,train_users=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36])
        INPUT_SIZE = 3
        CLASSES = 6
        SPLIT = 9
    else:
        print("Wrong Dataset Selected")
        sys.exit(0)

    for BATCH in BATCH_LIST:
        print("\nBATCH: " + str(BATCH))

        for LR in LR_LIST:
            print("\nLR: " + str(LR))

            logo = GroupKFold(n_splits=SPLIT)
            

            model_cpc = ContrastivePredictiveCoding(in_size=INPUT_SIZE)
            Path(output_dir).mkdir(exist_ok=True, parents=True)
            if Path(output_dir + "/cpc_start.pth").exists():
                model_cpc.load_state_dict(torch.load(output_dir + f"/{DATASET}__{exp_id}_cpc_start.pth"))
            else:
                torch.save(model_cpc.state_dict(), output_dir + f"/{DATASET}_{exp_id}_cpc_start.pth")

            model_cpc.to(device)
            params = model_cpc.parameters()
            optimizer = optim.SGD(params, lr=LR, momentum=0.9)

            map_losses = defaultdict(list)
            best_eval_loss = float('inf')
            curr_eval_patience = PATIENCE

            dir = output_dir + "/"+DATASET+"/"+str(exp_id)
            Path(dir).mkdir(exist_ok=True,parents=True)

            for epoch in range(NUM_EPOCHS):
                if epoch % 2 == 0:
                    for train_indices, val_indices in logo.split(ds, groups=ds.user_id_list):
                        user_id_array = np.array(ds.user_id_list)
                        train_user_ids = set(user_id_array[train_indices])
                        val_user_ids = set(user_id_array[val_indices])
                       
                        if np.random.uniform() > 0.5:
                            break
                    print("Train User IDs:", train_user_ids)
                    print("Validation User IDs:", val_user_ids)

                

                train_loader = DataLoader(dataset=Subset(ds, train_indices), batch_size=BATCH, shuffle=True)
                val_loader = DataLoader(dataset=Subset(ds, val_indices), batch_size=BATCH, shuffle=False)

                model_cpc.train()
                running_loss = 0.0

                for x, _ in train_loader:
                    optimizer.zero_grad()
                    x_orig = x.to(device)
                    loss = model_cpc.get_loss(x_orig)

                    running_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                running_loss /= len(train_loader)
                print(f"Epoch: {epoch}\tTraining loss: {running_loss:6f}\t", end='')
                map_losses['train'].append(running_loss)

                # Eval
                model_cpc.eval()
                loss_eval = 0.0
                with torch.no_grad():
                    for x, _ in val_loader:
                        x_orig = x.to(device)
                        loss = model_cpc.get_loss(x_orig)
                        loss_eval += loss.item()

                loss_eval /= len(val_loader)
                
                map_losses['val'].append(loss_eval)

                if loss_eval < best_eval_loss:
                    curr_eval_patience = PATIENCE
                    best_eval_loss = loss_eval

                    map_losses['best_loss'] = loss_eval
                    map_losses['best_epoch'] = epoch
                    torch.save(model_cpc.encoder.state_dict(), output_dir + f"/{DATASET}/{exp_id}/encoder.pth")
                    torch.save(model_cpc.state_dict(), output_dir + f"/{DATASET}/{exp_id}/cpc.pth")
                else:
                    curr_eval_patience -= 1
                if curr_eval_patience <= 0:
                    print("stop training as we are out of patience")
                    break
                print(f"eval loss: {loss_eval:6f} Patience: {curr_eval_patience}")

            # results
            with open(output_dir + f"/{DATASET}/{exp_id}/map_losses.json", 'w') as f_res:
                # add other hyper-params
                json.dump(map_losses, f_res)
            exp_id = exp_id + 1
