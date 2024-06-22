import numpy as np
import torch
from torch.utils.data import  DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
import sys
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from tqdm import tqdm
import os
import csv

from datasets.pamap2 import PAMAP2
from datasets.opportunity import Opportunity
from datasets.mmfit import MMFit
from datasets.mhealth import MHEALTH
from datasets.motionsense import MotionSense
from datasets.wisdm import WISDM

from models.cnn import CNN
from models.convlstm import ConvLSTM
from models.gru import GRU
from models.lstm import LSTM
from models.transformer import Transformer
from models.cpc import ClassificationWithEncoderCPC


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(device)

if 'SLURM_JOB_ID' in os.environ:
    outdir = '/netscratch/geissler/BeyondConfusion/outputs/'
else:
    outdir = 'outputs/'


if len(sys.argv) > 2:
    dataset_string = sys.argv[1]
    model_string = sys.argv[2]

    DATASET_LIST = [dataset_string]
    MODEL_LIST = [model_string]
else:
    # DATASET_LIST = ['PAMAP2','OPPORTUNITY','MMFIT']
    # MODEL_LIST = ['CPC', 'CNN','CONVLSTM','GRU','LSTM','TRANSFORMER']
    DATASET_LIST = ['MMFIT']
    MODEL_LIST = ['TRANSFORMER']

# BATCH_LIST = [64, 256, 1024]
# LR_LIST = [0.1, 0.01, 0.001]
BATCH_LIST = [256]
LR_LIST = [0.01]

WINDOW = 200
STRIDE = 100

STEP = 20
GAMMA = 0.5
PATIENCE = 40
NUM_EPOCHS = 200

# FOR TEST
# DATASET_LIST = ['OPPORTUNITY']
# MODEL_LIST = ['CNN']
# BATCH_LIST = [512]
# LR_LIST = [0.01,]
# PATIENCE = 40

exp_id = 0


# DATASET LOOP
for DATASET in DATASET_LIST:

    print("\nDATASET: " + DATASET)

    # Select Dataset
    if DATASET=='PAMAP2':
        ds = PAMAP2(users='full', window_size=WINDOW, window_step=STRIDE, frequency=50, columns=None, train_users=[1, 2, 3, 4, 5, 6, 7, 8])
        INPUT_SIZE = 27
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
        print(SPLIT)
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

    
    # MODEL LOOP
    for MODEL in MODEL_LIST:
        print("\nMODEL: " + MODEL)
        csv_file_path = outdir +DATASET+'_'+MODEL+'.csv'
        csvfile = open(csv_file_path, 'w', newline='')
        csv_writer = csv.writer(csvfile)

        for BATCH in BATCH_LIST:
            print("\nBATCH: " + str(BATCH))

            for LR in LR_LIST:
                print("\nLR: " + str(LR))
                
                logo = GroupKFold(n_splits=SPLIT)
                # CROSS-VALIDATION LOOP
                for train_indices, val_indices in logo.split(ds, groups=ds.user_id_list):
                    user_id_array = np.array(ds.user_id_list)
                    train_user_ids = set(user_id_array[train_indices])
                    val_user_ids = set(user_id_array[val_indices])

                    print("\nStart Fold")
                    print("Train User IDs:", train_user_ids)
                    print("Validation User IDs:", val_user_ids)

                    #Select Model
                    if MODEL=='CNN':
                        model = CNN(in_size=INPUT_SIZE, out_size=CLASSES)
                    elif MODEL=='CONVLSTM':   
                        model = ConvLSTM(in_size=INPUT_SIZE, out_size=CLASSES)
                    elif MODEL=='GRU':   
                        model = GRU(input_dim=INPUT_SIZE, output_dim=CLASSES, window_size=WINDOW)
                    elif MODEL=='LSTM':
                        model = LSTM(input_dim=INPUT_SIZE, output_dim=CLASSES, window_size=WINDOW)
                    elif MODEL=='TRANSFORMER':
                        model = Transformer(in_size=INPUT_SIZE, out_size=CLASSES, win_size=WINDOW)
                    elif MODEL == 'CPC':
                        model = ClassificationWithEncoderCPC(in_size=INPUT_SIZE, out_size=CLASSES, encoder_weights_path=f"pretrain_best/{DATASET}_encoder.pth")
                    else:
                        print("Wrong Model Selected")
                        sys.exit(0)

                    model.to(device)
                    #print(model)

                    best_val_loss = float('inf')
                    best_accuracy = 0
                    best_epoch = 0

                    wait = 0

                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=LR)
                    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP, gamma=GAMMA)

                    train_loader = DataLoader(dataset=Subset(ds, train_indices), batch_size=BATCH, shuffle=True)
                    val_loader = DataLoader(dataset=Subset(ds, val_indices), batch_size=BATCH, shuffle=False)

                    for epoch in range(NUM_EPOCHS):
                        running_loss = 0.0
                        num_batches = 0
                        val_loss = 0.0

                        model.train()
                        # with tqdm(train_loader, unit="batch") as t_batch:
                        #    for i, data in enumerate(t_batch):
                        for i, data in enumerate(train_loader):
                            x, y = data
                            optimizer.zero_grad()
                            x = x.to(device)
                            y = y.to(device)
                            outputs = model(x)
                            y=y.to(torch.int64)
                            loss = criterion(outputs, y)
                            loss.backward()

                            optimizer.step()
                            running_loss += loss.item()
                            num_batches += 1
                        train_loss = running_loss / num_batches
                        
                        model.eval() 
                        with torch.no_grad(): 
                            num_batches = 0 
                            correct_predictions = 0
                            total_predictions = 0
                            all_x = []
                            all_y = []
                            all_outputs = []
                            for i, data in enumerate(val_loader):
                                x, y = data
                                all_x.append(x)
                                all_y.append(y)
                                x = x.to(device)
                                y = y.to(device)
                                outputs = model(x)
                                all_outputs.append(outputs.cpu())
                                
                                _, predicted = torch.max(outputs, 1)
                                correct_predictions += (predicted == y).sum().item()
                                total_predictions += y.size(0)
                                y=y.to(torch.int64)
                                loss = criterion(outputs, y)
                                val_loss += loss.item()
                                num_batches += 1
                            
                            val_loss /= num_batches
                            accuracy = correct_predictions / total_predictions

                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                best_accuracy = accuracy
                                best_epoch = epoch+1

                                wait = 0
                                print(f"Best validation loss: {best_val_loss}")
                                print(f"Saving best model for epoch: {epoch+1}")
                                torch.save({
                                    'epoch': epoch+1,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'loss': criterion,
                                    }, outdir +DATASET+'_'+MODEL+'_'+str(exp_id)+'.pth')
                                
                                all_x = np.array([tensor.numpy() for tensor in all_x], dtype=object)
                                all_y = np.array([tensor.numpy() for tensor in all_y], dtype=object)
                                all_outputs = np.array([tensor.numpy() for tensor in all_outputs], dtype=object)
                                np.savez(outdir +DATASET+'_'+MODEL+'_'+str(exp_id)+'.npz', y=np.array(all_y, dtype=object), outputs=np.array(all_outputs, dtype=object))
                            else:
                                wait += 1
                                if wait >= PATIENCE:
                                    print(f"Validation loss did not decrease for {PATIENCE} epochs. Stopping training.")
                                    break
                                    
                        print('Epoch %d loss: %.5f, lr: %.5f, val loss: %.5f, val accuracy: %.5f' % (epoch+1, train_loss, optimizer.param_groups[0]['lr'], val_loss, accuracy))
                        scheduler.step()

                    csv_writer.writerow([exp_id, DATASET, MODEL, BATCH, LR, list(val_user_ids), list(train_user_ids), best_epoch, best_accuracy, best_val_loss, min(val_indices), max(val_indices)])
                    csvfile.flush()
                    
                    torch.cuda.empty_cache()
                    exp_id+=1
            
        csvfile.close()