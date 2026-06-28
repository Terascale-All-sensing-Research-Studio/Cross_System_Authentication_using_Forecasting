import os
import numpy as np
import time
import argparse
import logging
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from cross_datasets import Forecast_FineTune
from models import PatchTST
from utils.tools import EarlyStopping, adjust_learning_rate
from torch.optim import lr_scheduler 


def get_argumets():
    """
        Parse arguments from command line
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_root', type=str, required=False, default="splited_data",help='')
    parser.add_argument('--log_out_root', type=str, required=False, default='finetune_log',help='')
    parser.add_argument('--ckp_in_root', type=str, required=False, default='checkpoints',help='')
    parser.add_argument('--ckp_out_root', type=str, required=False, default='finetune_ckp',help='')
    parser.add_argument('--data', type=str, required=False, default='Data_2',help='')
    parser.add_argument('--train_for', type=str, required=False, default='Quest1',help='')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--learning_rate', type=int, default=0.0001, help='optimizer learning rate')

    parser.add_argument('--seq_len', type=int, default=25, help='input sequence length of Informer encoder')
    parser.add_argument('--label_len', type=int, default=10, help='start token length of Informer decoder')
    parser.add_argument('--pred_len', type=int, default=40, help='prediction sequence length')
    parser.add_argument("--use_feat", type=str, default="all", choices=["all", "right_all", "right_traj"], help='')

    parser.add_argument('--n_heads', type=int, default=16, help='num of heads')

    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument("--log", "-l", action="store_true", help='use this flag to save log files')

    parser.add_argument('--enc_in', type=int, default=21, help='encoder input size')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')

    return parser.parse_args()


if __name__ =="__main__":

    args = get_argumets()
    num_feat = 23

    logger = logging.getLogger("finetune_{}_sl{}_ll{}_pl{}".format(
        args.train_for,
        str(args.seq_len).zfill(2), 
        str(args.label_len).zfill(2), 
        str(args.pred_len).zfill(2)
        ))
    logger.setLevel(logging.DEBUG)
    fmt = "[%(name)s][%(asctime)s][%(levelname)s] >>> %(message)s"

    formatter = logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S')

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if args.log:
        log_out_to = os.path.join(
            args.log_out_root,
            "PatchTST"
        )

        if not os.path.exists(log_out_to):
            os.makedirs(log_out_to)

        fh = logging.FileHandler(
            os.path.join(
                log_out_to,
                "{}_sl{}_ll{}_pl{}.log".format(
                    args.train_for,
                    args.seq_len,
                    args.label_len,
                    args.pred_len
                )   
            ), 
            'w'
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.debug("Features:             {}".format(args.use_feat))
    logger.debug("Data:                 {}".format(args.data))
    logger.debug("Train for:            {}".format(args.train_for))
    logger.debug("Sequence length:      {}".format(args.seq_len))
    logger.debug("Label length:         {}".format(args.label_len))
    logger.debug("Prediction length:    {}".format(args.pred_len))
    logger.debug("Number of epoch:      {}".format(args.train_epochs))
    logger.debug("Batch size:           {}".format(args.batch_size))
    logger.debug("Learning rate:        {}".format(args.learning_rate))
    logger.debug("Dropout rate:         {}".format(args.dropout))

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    logger.info("Using device: {}\n".format(device))

    V3_train = Forecast_FineTune(
        root = args.data_root, 
        use_data=args.data,
        train_for=args.train_for,          
        forecasting_sizes=[args.seq_len, args.label_len, args.pred_len],
    )
    V3_train_loader = DataLoader(
        V3_train,
        batch_size=args.batch_size,
        shuffle=True
    )

    ckp_path = os.path.join(
       args.ckp_in_root, 
       "PatchTST_sl{}_ll{}_pl{}/checkpoint.pth".format(
            str(args.seq_len),
            str(args.label_len),
            str(args.pred_len)
       )
    )

    model = PatchTST.Model(args).float()
    model = model.to(device)
    model.load_state_dict(torch.load(ckp_path, map_location=device))

    ckp_out_path = os.path.join(
        args.ckp_out_root,
        "PatchTST",
        "{}_sl{}_ll{}_pl{}".format(
            args.train_for,
            args.seq_len,
            args.label_len,
            args.pred_len
        )
    )

    if not os.path.exists(ckp_out_path):
        os.makedirs(ckp_out_path)

    train_steps = len(V3_train_loader)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    scheduler = lr_scheduler.OneCycleLR(
        optimizer = model_optim,
        steps_per_epoch = train_steps,
        pct_start = args.pct_start,
        epochs = args.train_epochs,
        max_lr = args.learning_rate
    )

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()

        for i, (batch_x, batch_x_mark, batch_y, batch_y_mark, _) in enumerate(V3_train_loader):

            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)

            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            outputs = model(batch_x)

            f_dim = 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            loss.backward()
            model_optim.step()

        logger.info("Epoch: {} cost time: {:.4f}s".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        logger.info(
            "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss
            )
        )

        early_stopping(train_loss, model, ckp_out_path)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            logger.info(f"Save model to {ckp_out_path}")
            break

        adjust_learning_rate(model_optim, scheduler, epoch + 1, args, printout=False)

    logger.info("Training finished.")
    torch.cuda.empty_cache()