import os
import numpy as np
import tqdm
import argparse
import logging
from sklearn import metrics
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from cross_datasets import Authentication
from models import PatchTST


class FCN(nn.Module):
    def __init__(self, data_len, num_features=4, num_class=2):
        super(FCN, self).__init__()
        self.num_class = num_class

        self.c1 = nn.Conv1d(num_features, 128, kernel_size=8)
        self.bn1 = nn.BatchNorm1d(128)

        self.c2 = nn.Conv1d(128, 256, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(256)

        self.c3 = nn.Conv1d(256, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc = nn.Linear(data_len-13, num_class)

        self.relu = nn.ReLU()


    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.c1(x)        
        x = self.relu(self.bn1(x))

        x = self.c2(x)
        x = self.relu(self.bn2(x))

        x = self.c3(x)
        x = self.relu(self.bn3(x))
        x = x.transpose(1, 2)

        x = torch.mean(x, 2)
        x = self.fc(x.reshape(x.size()[0], -1)) 


        return x



def compute_eer(label, pred, positive_label=1):
   fpr, tpr, _ = metrics.roc_curve(y_true=label, y_score=pred, pos_label=positive_label)
   fnr = 1 - tpr

   eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
   eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

   eer = (eer_1 + eer_2) / 2
   return eer


def get_argumets():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_root', type=str, required=False, default="splited_data",help='')
    parser.add_argument('--out_root', type=str, required=False, default='output',help='')
    parser.add_argument('--ckp_in_root', type=str, required=False, default='PatchTST',help='')
    parser.add_argument('--data', type=str, required=False, default='Data_2',help='')
    parser.add_argument('--train_for', type=str, required=False, default='Quest1',help='')
    parser.add_argument('--test_for', type=str, required=False, default='Quest2',help='')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=int, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--max_epoch', type=int, default=250, help='')

    parser.add_argument('--seq_len', type=int, default=25, help='input sequence length of Informer encoder')
    parser.add_argument('--label_len', type=int, default=10, help='start token length of Informer decoder')
    parser.add_argument('--pred_len', type=int, default=40, help='prediction sequence length')
    parser.add_argument("--use_feat", type=str, default="all", choices=["all", "right_all", "right_traj"], help='')

    parser.add_argument('--num_class', type=int, default=2, help='number of classification classes')

    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')

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


if __name__ == "__main__":
    args = get_argumets()

    num_feat = 23

    for user_id in all_users:
        logger = logging.getLogger("Auth_{}_{}_userID{}_sl{}_ll{}_pl{}".format(
            args.train_for,
            args.test_for,
            str(user_id).zfill(2),
            str(args.seq_len).zfill(2), 
            str(args.label_len).zfill(2), 
            str(args.pred_len).zfill(2)
            ))
        logger.setLevel(logging.DEBUG)
        fmt = "[%(name)s] %(levelname)s>>> %(message)s"
        formatter = logging.Formatter(fmt)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if args.log:
            log_out_to = os.path.join(args.out_root, args.train_for, args.test_for)
            if not os.path.exists(log_out_to):
                os.makedirs(log_out_to)

            fh = logging.FileHandler(os.path.join(
                log_out_to,
                "Auth_{}_{}_userID{}_sl{}_ll{}_pl{}.log".format(
                    args.train_for,
                    args.test_for,
                    str(user_id).zfill(2),
                    str(args.seq_len).zfill(2),
                    str(args.label_len).zfill(2),
                    str(args.pred_len).zfill(2),
                    )
                ), 
                'w'
                )
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        logger.debug("User:                 {}".format(user_id))
        logger.debug("Features:             {}".format(args.use_feat))
        logger.debug("Train for:            {}".format(args.train_for))
        logger.debug("Test for:             {}".format(args.test_for))
        logger.debug("Sequence length:      {}".format(args.seq_len))
        logger.debug("Label length:         {}".format(args.label_len))
        logger.debug("Prediction length:    {}".format(args.pred_len))
        logger.debug("Number of epoch:      {}".format(args.max_epoch))
        logger.debug("Batch size:           {}".format(args.batch_size))
        logger.debug("Learning rate:        {}".format(args.learning_rate))
        logger.debug("Dropout rate:         {}".format(args.dropout))

        device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
        logger.info("Using device:          {}\n".format(device))
            
            
        auth_train_data = Authentication(
            root=args.data_root,
            use_data=args.train_for,
            user_id=user_id, 
            forecasting_sizes=[args.seq_len, args.label_len, args.pred_len],
        )
        auth_train_loader = DataLoader(
            auth_train_data,
            batch_size=args.batch_size,
            shuffle=True
        )
        
        auth_test_data = Authentication(
            root=args.data_root,
            use_data=args.test_for,
            user_id=user_id, 
            forecasting_sizes=[args.seq_len, args.label_len, args.pred_len],
        )
        auth_test_loader = DataLoader(
            auth_test_data,
            batch_size=args.batch_size,
            shuffle=False
        )

        logger.info("{} training data loaded!".format(len(auth_train_data)))
        logger.info("{} testing data loaded!".format(len(auth_test_data)))

        forecast_model = PatchTST.Model(args).float()
        forecast_model = forecast_model.to(device)

        ckp_path = os.path.join(
            args.ckp_in_root, 
            "{}_sl{}_ll{}_pl{}".format(
                args.train_for,
                args.seq_len,
                args.label_len, 
                args.pred_len
            ),
            "checkpoint.pth"
        )

        forecast_model.load_state_dict(torch.load(ckp_path, map_location=device))

        for params in forecast_model.parameters():
            params.requires_grad = False

        auth_model = FCN(
            data_len=args.seq_len + args.pred_len, 
            num_features=num_feat,
            num_class=args.num_class
        ).to(device)
  
        model_optim = optim.Adam(auth_model.parameters(), lr=args.learning_rate)
        criterion_bce_classification =  nn.BCELoss().to(device)

        checkpoint_path = os.path.join(args.out_root, args.train_for, "checkpoints")
        logger.info("checkpoint saved to {}".format(checkpoint_path))

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        global_eer = np.inf
        best_eer_epoch = 0
        for epoch in tqdm.tqdm(range(args.max_epoch)):
            epoch_loss = 0

            for i, (batch_x, batch_x_mark, batch_y, batch_y_mark, _, gt_label) in enumerate(auth_train_loader):

                model_optim.zero_grad()

                gt_label = torch.eye(2)[gt_label.long(), :].to(device)

                batch_x = batch_x.float().to(device)

                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

                with torch.no_grad():
                    forecast_model.eval()
                    forecasting_output = forecast_model(batch_x)

                auth_input = torch.concat((batch_x, forecasting_output), dim=1)

                auth_model.train()
                classification_output = auth_model(auth_input)

                mm = nn.Sigmoid()
                bce_class_loss = criterion_bce_classification(mm(classification_output), gt_label)
                epoch_loss += bce_class_loss.item()
                bce_class_loss.backward()
                model_optim.step()

            logger.info("Train Epoch {}/{} Auth loss: {}".format(
                epoch + 1, 
                args.max_epoch,
                epoch_loss / len(auth_train_loader)
            ))

            val_loss = 0
            correct_t = 0
            total_t = 0

            with torch.no_grad():
                forecast_model.eval()
                auth_model.eval()
                
                all_labels_t = []
                all_preds_t = []

                for i_t, (batch_x_t, batch_x_mark_t, batch_y_t, batch_y_mark_t, _, gt_label_t) in enumerate(auth_test_loader):
                    all_labels_t.append(gt_label_t.tolist())
                    labels_t1 = gt_label_t.to(device)

                    batch_x_t = batch_x_t.float().to(device)
                    batch_x_mark_t = batch_x_mark_t.float().to(device)
                    
                    dec_inp_t = torch.zeros([batch_y_t.shape[0], args.pred_len, batch_y_t.shape[-1]]).float()
                    batch_y_t = torch.cat([batch_y_t[:,:args.label_len,:], dec_inp_t], dim=1).float().to(device)
                    batch_y_mark_t = batch_y_mark_t.float().to(device)

                    forecasting_output_t = forecast_model(batch_x_t)
                    
                    auth_input_t = torch.concat((batch_x_t, forecasting_output_t), dim=1)
                    
                    classification_output_t = auth_model(auth_input_t)
                    _, pred_t = torch.max(classification_output_t, dim=1)
                    all_preds_t.append(pred_t.cpu().tolist())
                    
                    correct_t += torch.sum(pred_t==labels_t1).item()
                    total_t += gt_label_t.size(0)

                flatten_list = lambda y:[x for a in y for x in flatten_list(a)] if type(y) is list else [y]
                all_labels_t = flatten_list(all_labels_t)
                all_preds_t = flatten_list(all_preds_t)
                eer = compute_eer(all_labels_t, all_preds_t)

                if eer < global_eer:
                    global_eer = eer
                    best_eer_epoch = epoch

                    torch.save(
                        auth_model.state_dict(), 
                        os.path.join(
                            checkpoint_path,
                            "Auth_{}_{}_userID{}_sl{}_ll{}_pl{}.pth".format(
                                args.train_for,
                                args.test_for,
                                str(user_id).zfill(2),
                                str(args.seq_len).zfill(2),
                                str(args.label_len).zfill(2),
                                str(args.pred_len).zfill(2)
                            )
                        )
                    )

                logger.info("Testing using {} in epoch {}/{} with ACC {} and EER {}".format(
                    args.test_for, 
                    epoch+1, 
                    args.max_epoch, 
                    correct_t / total_t,
                    eer
                ))
    
        logger.info("Best EER is {} from epoch {}".format(global_eer, best_eer_epoch))
        logger.info("Training finished.")
        torch.cuda.empty_cache()