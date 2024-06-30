import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import time
import torch.optim as optim
import pickle
import torch
import torch.nn as nn
import CNN_TimesNet_cga as tn
import csv
from datetime import datetime

pd.set_option('display.width', 300)
pd.set_option('display.max_columns', None)

class Config:
    def __init__(self):
        self.num_epochs = 1
        self.batch_size = 64
        self.learning_rate = 0.005
        self.seq_len = 48
        self.pred_len = 1
        self.top_k = 3
        self.d_model = 128
        self.d_ff = 256
        self.num_kernels = 2
        self.embed = 'timeF'
        self.enc_in = 3200
        self.freq = 'h'
        self.dropout = 0
        self.e_layers = 2
        self.c_out = 3200

config = Config()

def Standard_X(nparray):
    shape = nparray.shape
    nparray_1 = nparray.reshape(-1, 2)
    mean = np.mean(nparray_1, axis=0)
    std = np.std(nparray_1, axis=0)
    nparray_std = (nparray_1 - mean) / std
    nparray = nparray_std.reshape(shape)
    return nparray,mean,std
def Standard_Y(nparray, mean, std):
    shape = nparray.shape
    nparray_1 = nparray.reshape(-1, 2)
    nparray_std = (nparray_1 - mean) / std
    nparray = nparray_std.reshape(shape)
    return nparray
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        mse = torch.mean((y_pred - y_true) ** 2)
        rmse = torch.sqrt(mse)
        return rmse
class MultiLosses(nn.Module):
    def __init__(self):
        super(MultiLosses, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.rmse_loss = RMSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, y_pred, y_true):
        mse = self.mse_loss(y_pred, y_true)
        rmse = self.rmse_loss(y_pred, y_true)
        mae = self.mae_loss(y_pred, y_true)
        return mse, rmse, mae
def calculate_losses(output_list, target_list, criterion):
    loss_list = []
    for i in range(len(output_list)):
        losses = criterion(output_list[i], target_list[i])
        loss_dict = {'mse': losses[0].item(), 'rmse': losses[1].item(), 'mae': losses[2].item()}
        loss_list.append(loss_dict)
    return loss_list

def save_loss_list_to_csv(loss_list, csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Sample', 'MSE', 'RMSE', 'MAE'])
        for idx, loss in enumerate(loss_list):
            writer.writerow([idx, loss['mse'], loss['rmse'], loss['mae']])
def match_time(time, dataframe):
    time_dict = {}
    for time_str in time:
        time_obj = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        for i, row in dataframe.iterrows():
            time_data_obj = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
            if time_data_obj.date() == time_obj.date() and time_data_obj.hour == time_obj.hour:
                time_dict[time_str] = np.array(row[1:])
                break
    return time_dict
def train_model(i, model,
                X_train, Y_train, X_val, Y_val, num_epochs=config.num_epochs, batch_size=config.batch_size, learning_rate=config.learning_rate):
    since = time.time()
    criterion = nn.MSELoss()
    mae_loss = nn.L1Loss()
    rmse_loss = RMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.2)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.float32).reshape(-1, 1, 3200)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).reshape(-1, 1, 3200)

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    seq_str = str(config.seq_len)
    location = str(i)
    csv_name = 'output/' + seq_str + '_' + location + '_loss.csv'
    model_name = 'output/' + seq_str + '_' + location + '_best_model.pth'
    holiday_loss_csv = 'output/' + seq_str + '_' + location + '_holidayloss.csv'
    workday_loss_csv = 'output/' + seq_str + '_' + location + '_workdayloss.csv'
    peak_loss_csv = 'output/' + seq_str + '_' + location + '_peakloss.csv'
    holiday_pkl = 'output/' + seq_str + '_' + location + '_holiday.pkl'
    workday_pkl = 'output/' + seq_str + '_' + location + '_workday.pkl'
    peak_pkl = 'output/' + seq_str + '_' + location + '_peak.pkl'

    with open(csv_name, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Epoch', 'Train  Loss  (MSE)', 'Val  Loss  (MSE)', 'Train  Loss  (MAE)',
                         'Val  Loss  (MAE)', 'Train  Loss  (RMSE)', 'Val  Loss  (RMSE)'])

    best_val_loss = float('inf')
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_mae_loss = 0.0
        total_rmse_loss = 0.0
        holiday_output_list = []
        holiday_Y_list = []
        workday_output_list = []
        workday_Y_list = []
        peak_output_list = []
        peak_Y_list = []

        for batch_X, batch_Y in train_data_loader:
            time_list = batch_X[:, :time_data_shape]
            if torch.cuda.is_available():
                batch_X = batch_X.cuda()
                batch_Y = batch_Y.cuda()

            output = model(batch_X).to(device)
            output = output.reshape(batch_Y.size())

            for idx, time_info in enumerate(time_list):
                if time_info[2] == 1:
                    holiday_output_list.append(output[idx].squeeze())
                    holiday_Y_list.append(batch_Y[idx].squeeze())
                elif time_info[2] == 0:
                    workday_output_list.append(output[idx].squeeze())
                    workday_Y_list.append(batch_Y[idx].squeeze())
            for idx, time_info in enumerate(time_list):
                if time_info[4] == 1:
                    peak_output_list.append(output[idx].squeeze())
                    peak_Y_list.append(batch_Y[idx].squeeze())

            loss = criterion(output, batch_Y)
            mae_train_loss = mae_loss(output, batch_Y)
            rmse_train_loss = rmse_loss(output, batch_Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            total_loss += loss.item()
            total_mae_loss += mae_train_loss.item()
            total_rmse_loss += rmse_train_loss.item()

        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            mae_val_loss = 0.0
            rmse_val_loss = 0.0
            for batch_X_val, batch_Y_val in val_data_loader:
                if torch.cuda.is_available():
                    batch_X_val = batch_X_val.cuda()
                    batch_Y_val = batch_Y_val.cuda()

                output_val = model(batch_X_val)
                output_val = output_val.reshape(batch_Y_val.size())
                val_loss += criterion(output_val, batch_Y_val).item()
                mae_val_loss += mae_loss(output_val, batch_Y_val).item()
                rmse_val_loss += rmse_loss(output_val, batch_Y_val).item()

            train_loss_avg = total_loss / len(train_data_loader)
            mae_train_loss_avg = total_mae_loss / len(train_data_loader)
            rmse_train_loss_avg = total_rmse_loss / len(train_data_loader)
            val_loss_avg = val_loss / len(val_data_loader)
            mae_val_loss_avg = mae_val_loss / len(val_data_loader)
            rmse_val_loss_avg = rmse_val_loss / len(val_data_loader)
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss_avg}, Val Loss: {val_loss_avg}")
            scheduler.step()

            with open(csv_name, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, train_loss_avg, val_loss_avg, mae_train_loss_avg,
                                 mae_val_loss_avg, rmse_train_loss_avg, rmse_val_loss_avg])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()

                criterion1 = MultiLosses()

                holiday_losses = calculate_losses(holiday_output_list, holiday_Y_list, criterion1)
                workday_losses = calculate_losses(workday_output_list, workday_Y_list, criterion1)
                peak_losses = calculate_losses(peak_output_list, peak_Y_list, criterion1)

                save_loss_list_to_csv(holiday_losses, holiday_loss_csv)
                save_loss_list_to_csv(workday_losses, workday_loss_csv)
                save_loss_list_to_csv(peak_losses, peak_loss_csv)

                with open(holiday_pkl, 'wb') as file:
                    pickle.dump([torch.stack(holiday_output_list), torch.stack(holiday_Y_list)], file)

                with open(workday_pkl, 'wb') as file:
                    pickle.dump([torch.stack(workday_output_list), torch.stack(workday_Y_list)], file)

                with open(peak_pkl, 'wb') as file:
                    pickle.dump([torch.stack(peak_output_list), torch.stack(peak_Y_list)], file)

        torch.save(best_model, model_name)

        time_elapsed = time.time() - since
        print(seq_str + '_' + location + '_Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                                                        time_elapsed % 60))


seq_len1 = config.seq_len
seq1 = str(seq_len1)
for i in range(4): 
    m = str(i)
    for j in range(1, 3):
        n = str(j)
        flie_name = r'data/travel_dict_区域' + m + '_时间' + n + '.pkl'
        flie_shape = r'data/travel_list_shape.pkl'
        with open(flie_name, 'rb') as f:
            travel_dict = pickle.load(f)
        with open(flie_shape, 'rb') as f:
            travel_list_shape = pickle.load(f)

        file_name = f'data/resampling_data_区域{m}_时间{n}.pkl'
        with open(file_name, 'rb') as f:
            X_train, Y_train, X_val, Y_val, time_train, time_val = pickle.load(f)

        if j == 1:
            X_train1 = X_train
            Y_train1 = Y_train
            time_train1 = time_train
            X_val1 = X_val
            Y_val1 = Y_val
            time_val1 = time_val
        else:
            X_train = np.concatenate((X_train1, X_train), axis=0)
            Y_train = np.concatenate((Y_train1, Y_train), axis=0)
            time_train = np.concatenate((time_train1, time_train), axis=0)
            X_val = np.concatenate((X_val1, X_val), axis=0)
            Y_val = np.concatenate((Y_val1, Y_val), axis=0)
            time_val = np.concatenate((time_val1, time_val), axis=0)

            datalength_X_train = X_train.shape
            datalength_X_train = datalength_X_train[0]
            X_train = X_train.reshape((datalength_X_train,) + travel_list_shape).transpose(0, 1, 4, 2, 3)
            Y_train = Y_train.reshape(travel_list_shape).transpose(0, 3, 1, 2)
            X_train = X_train.reshape(datalength_X_train, -1)
            datalength_X_val = X_val.shape
            datalength_X_val = datalength_X_val[0]
            X_val = X_val.reshape((datalength_X_val,) + travel_list_shape).transpose(0, 1, 4, 2, 3)
            Y_val = Y_val.reshape(travel_list_shape).transpose(0, 3, 1, 2)
            X_val = X_val.reshape(datalength_X_val, -1)

            if i < 2:
                time_file = 'data/beijing.csv'
            else:
                time_file = 'data/shenzhen.csv'
            time_data = pd.read_csv(time_file, encoding='utf-8')
            time_train_data = match_time(time_train, time_data)
            time_train_data = np.array(list(time_train_data.values()), dtype=np.float32)
            time_val_data = match_time(time_val, time_data)
            time_val_data = np.array(list(time_val_data.values()), dtype=np.float32)

            time_data_shape = time_train_data.shape
            time_data_shape = time_data_shape[1]

            X_train = np.concatenate((time_train_data, X_train), axis=1)
            X_val = np.concatenate((time_val_data, X_val), axis=1)

            train_on_gpu = torch.cuda.is_available()
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            model = tn.Model(seq_len1, pred_len = config.pred_len, top_k = config.top_k, d_model = config.d_model, d_ff = config.d_ff, num_kernels = config.num_kernels,embed = config.embed,
                             enc_in = config.enc_in, freq = config.freq, dropout = config.dropout, e_layers = config.e_layers,  c_out = config.c_out, time_data_shape =time_data_shape)

            train_model(i, model, X_train, Y_train, X_val, Y_val)
            print("Training finished!")




