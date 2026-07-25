"""
train.py  (adapted for the ABCD simulations)

The training algorithm, model selection, early stopping, and returned values
are UNCHANGED from the original DKNN implementation. The only differences are
practical for batch runs over 4 settings x 50 replicates x 2 coordinates:
  * The very verbose per-step / describe() prints are gated behind `if_summary`
    so a full sweep does not emit hundreds of thousands of log lines.
  * `epoch` is exposed as a keyword argument (default 300, the original value).

Note (state plainly in the paper): DKNN selects its best epoch on the
evaluation (test) split, as in the authors' original code. This is kept as-is
(changing it would alter the method); it makes DKNN's point metrics, if
anything, mildly optimistic — i.e. the baseline is treated generously.
"""
import datetime, os
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils import DIAGNOSIS, EarlyStopping, RMSE, MSE, MAE, MAPE
from utils.loaddataset import DataSet
from model.net import DKNN


def train(modelname, datapath,
          batch_size=128, lr=0.0001, hidden_neurons=[4, 256, 16], pe_weight=0.8, top_k=16,
          loss_type='rmse', optim_type='adam', if_summary=False, if_save_model=False,
          epoch=300):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    ##### read data #####
    data = pd.read_csv(datapath)
    if if_summary:
        print(data.head())

    ##### process data #####
    dataset = DataSet(data)
    data_scaler = dataset.scaler_data()
    data_train_scaler = data_scaler['train']
    data_test_scaler = data_scaler['test']
    if if_summary:
        print('data_train_scaler:')
        print(data_train_scaler.describe())

    train_dataloader = DataLoader(data_train_scaler.values.astype(float),
                                  shuffle=True, batch_size=batch_size, drop_last=False)
    test_dataloader = DataLoader(data_test_scaler.values.astype(float),
                                 shuffle=False, batch_size=batch_size, drop_last=False)

    # Take the train set as known points (observed locations)
    known_coods_scaler = data_train_scaler.values[:, 1:3]
    known_feature_scaler = data_train_scaler.values[:, 3:]
    known_coods_scaler = torch.from_numpy(known_coods_scaler.astype(float)).to(torch.float32).to(device)
    known_feature_scaler = torch.from_numpy(known_feature_scaler.astype(float)).to(torch.float32).to(device)

    ##### DKNN model initialization #####
    modelname = 'DKNN'
    d_input, d_model, d_trend = hidden_neurons
    model = DKNN(d_input=d_input, d_model=d_model, known_num=dataset.train_num,
                 d_trend=d_trend, top_k=top_k, pe_weight=pe_weight)
    model.cal_pe_know(known_feature_scaler, known_coods_scaler)
    model.cal_pe_unknow(
        torch.from_numpy(data_test_scaler.values[:, 3:3 + d_input].astype(float)).to(torch.float32).to(device),
        torch.from_numpy(data_test_scaler.values[:, 1:3].astype(float)).to(torch.float32).to(device))
    model.to(device)

    ##### loss function #####
    if loss_type == 'mae':
        criterion = MAE()
    elif loss_type == 'mse':
        criterion = MSE()
    elif loss_type == 'rmse':
        criterion = RMSE()
    elif loss_type == 'mape':
        criterion = MAPE()

    ##### optimizer #####
    if optim_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    elif optim_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8,
                                                     verbose=False, patience=5, min_lr=1e-6)

    ##### save path / summary #####
    datafile = datapath[datapath.rfind('/') + 1:]
    datafilename = datafile[0:datafile.rfind('.')]
    start_time = datetime.datetime.now()
    train_info = start_time.strftime("%m%d_%H%M%S_%f")     # %f avoids collisions in fast loops
    save_dir = './results/' + modelname + '/' + datafilename + '/' + train_info

    if if_summary:
        summarypath = save_dir + '/summary/'
        writer = SummaryWriter(summarypath)
        print('summary path: \n' + summarypath)

    ##### early stopping #####
    patience = 30
    early_stopping = EarlyStopping(patience, verbose=False)

    ##### start train #####
    total_train_step = 0
    min_loss, best_epoch = float('inf'), float('inf')
    best_rmse_inverse, best_mae_inverse, best_mape_inverse = float('inf'), float('inf'), float('inf')

    for e in tqdm(range(epoch), disable=not if_summary):

        ##### training #####
        total_train_loss = 0
        train_num = 0
        model.train()
        for step, i in enumerate(train_dataloader):
            i = i.to(torch.float32)
            input_pe = model.pe_know[i[:, 0].type(torch.long)]
            input_coods = i[:, 1:3].to(device)
            input_feature = i[:, 3:3 + d_input].to(device)
            input_feature[:, -1] = 0

            optimizer.zero_grad()
            output, _ = model(input_coods, input_feature, input_pe,
                              known_coods_scaler, known_feature_scaler)

            target = i[:, -1].to(device)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            diag = DIAGNOSIS(output, target)
            l_rmse, l_mse, l_mae, l_mape = diag.get()

            if if_summary:
                total_train_step += 1
                writer.add_scalar("train_loss", loss, total_train_step)
                writer.add_scalar("train_rmse", l_rmse, total_train_step)
                writer.add_scalar("train_mae", l_mae, total_train_step)
                writer.add_scalar("train_mape", l_mape, total_train_step)
                if step % 3 == 0:
                    print('\n--------train epoch {},step {},loss: {:.5f}, RMSE:{:.5f},MSE:{:.5f}, MAE:{:.5f}, MAPE:{:.5f}--'
                          .format(e, step, loss.item(), l_rmse, l_mse, l_mae, l_mape))

            total_train_loss += loss.item()
            train_num += 1

        total_train_loss = total_train_loss / train_num
        if if_summary:
            writer.add_scalar("total_train_loss", total_train_loss, e)

        ##### testing #####
        total_test_loss = 0
        l2p, l3p, l4p = 0, 0, 0
        lrmse_inverse, lmae_inverse, lmape_inverse = 0, 0, 0
        test_num = 0
        with torch.no_grad():
            model.eval()
            for step, i in enumerate(test_dataloader):
                i = i.to(torch.float32)
                input_pe = model.pe_unknow[i[:, 0].type(torch.long)]
                input_coods = i[:, 1:3].to(device)
                input_feature = i[:, 3:3 + d_input].to(device)
                input_feature[:, -1] = 0

                output, _ = model(input_coods, input_feature, input_pe,
                                  known_coods_scaler, known_feature_scaler)

                target = i[:, -1].to(device)
                test_loss = criterion(output, target)
                test_diag = DIAGNOSIS(output, target)
                t_rmse, t_mse, _, t_mape = test_diag.get()
                total_test_loss += test_loss.item()
                l2p += t_rmse
                l3p += t_mse
                l4p += t_mape

                output_inverse = dataset.scaler_label.inverse_transform(output.cpu().reshape(-1, 1))
                target_inverse = dataset.scaler_label.inverse_transform(target.cpu().reshape(-1, 1))
                test_diag_inverse = DIAGNOSIS(output_inverse, target_inverse)
                lrmse_inverse += test_diag_inverse.v_rmse
                lmae_inverse += test_diag_inverse.v_mae
                lmape_inverse += test_diag_inverse.v_mape

                test_num += 1

            final_loss = total_test_loss / test_num
            final_rmse = l2p / test_num
            final_mse = l3p / test_num
            final_mape = l4p / test_num
            final_rmse_inverse = lrmse_inverse / test_num
            final_mae_inverse = lmae_inverse / test_num
            final_mape_inverse = lmape_inverse / test_num

            scheduler.step(final_loss)

            if if_summary:
                writer.add_scalar("total_test_loss", final_loss, e)
                writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], e)
                end_time = datetime.datetime.now()
                print('--------test loss: {:.5f}, RMSE:{:.5f},MSE:{:.5f}, MAPE:{:.5f}, RMSE_inv:{:.5f}, epoch time:{:.2f}m--'
                      .format(final_loss, final_rmse, final_mse, final_mape, final_rmse_inverse,
                              (end_time - start_time).seconds / 60))

            # save the best model (selected on the evaluation split, as in original)
            if final_loss < min_loss:
                min_loss = final_loss
                best_epoch = e
                best_rmse_inverse = final_rmse_inverse
                best_mae_inverse = final_mae_inverse
                best_mape_inverse = final_mape_inverse

                if if_save_model:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(model.state_dict(), save_dir + '/checkpoint.pth')
                    with open(save_dir + '/checkpoint_info.txt', 'w') as f:
                        f.write('-----------model_info-------------')
                        f.write('\rmodel name:  dknn\r')
                        f.write('train info:  {}\r'.format(train_info))
                        f.write('min loss:  {}\r'.format(min_loss))
                        f.write('best_epoch:  {}\r'.format(best_epoch))
                        f.write('best_rmse_inverse:  {}\r'.format(best_rmse_inverse))
                        f.write('best inverse MAE/RMSE/MAPE: {:.2f}/{:.2f}/{:.2f}%\r'
                                .format(best_mae_inverse, best_rmse_inverse, best_mape_inverse * 100))
                        f.write('epoch: {}\r'.format(e))

            early_stopping(final_loss, model, save_model_dir=None)
            if early_stopping.early_stop:
                break

    if if_summary:
        writer.close()
        print('******** {} *********'.format(datafilename))
        print('final min loss:', min_loss, '| best_epoch:', best_epoch)
        print('best inverse MAE/RMSE/MAPE: {:.2f}/{:.2f}/{:.2f}%'
              .format(best_mae_inverse, best_rmse_inverse, best_mape_inverse * 100))

    best_inverse = [best_mae_inverse, best_rmse_inverse, best_mape_inverse]
    return train_info, min_loss, best_epoch, best_inverse
