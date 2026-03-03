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

from data import Authentication
from models.main_model import V3_Forecast_Model
from models.MJ_FCN import FCN

all_users = [5, 6, 10, 11, 12, 13, 15, 17, 18, 20, 21, 22,
             23, 27, 31, 32, 35, 37, 38, 40]

def compute_eer(label, pred, positive_label=1):
   # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
   fpr, tpr, threshold = metrics.roc_curve(y_true=label, y_score=pred, pos_label=positive_label)
   fnr = 1 - tpr

   # the threshold of fnr == fpr
   eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

   # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
   eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
   eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

   # return the mean of eer from fpr and from fnr
   eer = (eer_1 + eer_2) / 2
   return eer

def get_argumets():
    """
        Parse arguments from command line
    """
    parser = argparse.ArgumentParser(description='cross system VR auth')
    parser.add_argument('--data_root', type=str, required=False, default="splited_data",help='')
    parser.add_argument('--out_root', type=str, required=False, default='authentication',help='')
    parser.add_argument('--ckp_in_root', type=str, required=False, default='finetune',help='')
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

    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument("--log", "-l", action="store_true", help='use this flag to save log files')

    return parser.parse_args()


if __name__ =="__main__":

    args = get_argumets()

    if args.use_feat == "right_traj":
        num_feat = 4
    elif args.use_feat == "right_all":
        num_feat = 8
    else:
        num_feat = 23
        
    for user_id in all_users:
        logger = logging.getLogger("V3 Auth_{}_{}_userID{}_sl{}_ll{}_pl{}".format(
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
            
            
        # load training data
        auth_train_data = Authentication(
            root=args.data_root,
            use_data=args.train_for,
            user_id=user_id, 
            forecasting_sizes=[args.seq_len, args.label_len, args.pred_len],
            use_feat="all"
        )
        auth_train_loader = DataLoader(
            auth_train_data,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False
        )
        
        # load testing data
        auth_test_data = Authentication(
            root=args.data_root,
            use_data=args.test_for,
            user_id=user_id, 
            forecasting_sizes=[args.seq_len, args.label_len, args.pred_len],
            use_feat="all"
        )
        auth_test_loader = DataLoader(
            auth_test_data,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True
        )

        # load forecasting model
        forecast_model = V3_Forecast_Model(
            enc_in=num_feat, dec_in=num_feat, c_out=num_feat, 
            seq_len=args.seq_len, label_len=args.label_len, out_len=args.pred_len,
            num_feats=num_feat, d_model=args.d_model, n_heads=args.n_heads,
            e_layers=args.e_layers, d_layers=args.d_layers, 
            d_ff=args.d_ff, dropout=args.dropout, activation='gelu', output_attention=False,
            device=device
        ).to(device)
        
        ckp_path = os.path.join(
            args.ckp_in_root, 
            args.train_for, 
            "all_checkpoints", 
            "finetune_test_{}_{}_all_sl{}_ll{}_pl{}.pth".format(
                args.train_for,
                args.test_for,
                str(args.seq_len).zfill(2),
                str(args.label_len).zfill(2),
                str(args.pred_len).zfill(2)
            )
        )
        forecast_model.load_state_dict(torch.load(ckp_path, map_location=device))
 
        for params in forecast_model.parameters():
            params.requires_grad = False

        auth_model = FCN(
            data_len=args.seq_len + args.pred_len, 
            num_features=num_feat,
            num_class=args.num_class
        ).to(device)

        # define optimizer and loss function    
        model_optim = optim.Adam(auth_model.parameters(), lr=args.learning_rate)
        criterion_bce_classification =  nn.BCELoss().to(device)
        
        # checkpoints out
        checkpoint_path = os.path.join(args.out_root, args.train_for, f"{args.use_feat}_checkpoints")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
            
        global_eer = np.inf
        best_eer_epoch = 0
        for epoch in tqdm.tqdm(range(args.max_epoch)):
            ''' training using args.train_for '''
            epoch_loss = 0
            for i, (en_in, en_time, de_in, de_time, _, gt_label) in enumerate(auth_train_loader):
                model_optim.zero_grad()
                gt_label = torch.eye(2)[gt_label.long(), :].to(device)
                
                en_in = en_in.float().to(device)
                en_time = en_time.float().to(device)
                
                dec_inp = torch.zeros([de_in.shape[0], args.pred_len, de_in.shape[-1]]).float()
                de_in = torch.cat([de_in[:,:args.label_len,:], dec_inp], dim=1).float().to(device)
                de_time = de_time.float().to(device)

                # forecasting
                with torch.no_grad():
                    forecast_model.eval()
                    _, forecasting_output = forecast_model(en_in, en_time, de_in, de_time)
                
                auth_input = torch.concat((en_in, forecasting_output), dim=1)
                
                # Authentication
                auth_model.train()
                classification_output = auth_model(auth_input)
                
                # compute loss
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
            
            ''' testing using args.test_for '''
            val_loss = 0
            correct_t = 0
            total_t = 0
            with torch.no_grad():
                forecast_model.eval()
                auth_model.eval()
                
                all_labels_t = []
                all_preds_t = []
                for i_t, (en_in_t, en_time_t, de_in_t, de_time_t, _, gt_label_t) in enumerate(auth_test_loader):
                    all_labels_t.append(gt_label_t.tolist())
                    labels_t1 = gt_label_t.to(device)
                    en_in_t = en_in_t.float().to(device)
                    en_time_t = en_time_t.float().to(device)
                    
                    dec_inp_t = torch.zeros([de_in_t.shape[0], args.pred_len, de_in_t.shape[-1]]).float()
                    de_in_t = torch.cat([de_in_t[:,:args.label_len,:], dec_inp_t], dim=1).float().to(device)
                    de_time_t = de_time_t.float().to(device)
                    
                    _, forecasting_output_t = forecast_model(en_in_t, en_time_t, de_in_t, de_time_t)
                    
                    auth_input_t = torch.concat((en_in_t, forecasting_output_t), dim=1)
                    classification_output_t = auth_model(auth_input_t)
                    _, pred_t = torch.max(classification_output_t, dim=1)
                    all_preds_t.append(pred_t.cpu().tolist())
                    
                    # ACC
                    correct_t += torch.sum(pred_t==labels_t1).item()
                    total_t += gt_label_t.size(0)
                
                flatten_list = lambda y:[x for a in y for x in flatten_list(a)] if type(y) is list else [y]
                all_labels_t = flatten_list(all_labels_t)
                all_preds_t = flatten_list(all_preds_t)
                eer = compute_eer(all_labels_t, all_preds_t)
                
                if eer < global_eer:
                    global_eer = eer
                    best_eer_epoch = epoch
                    
                    # save the best testing auth model
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