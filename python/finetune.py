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

from data import Forecast_FineTune
from models.main_model import V3_Forecast_Model


def validation(
        model,
        test_data_name,
        data_loader,
        global_loss,
        device,
        args,
    ):
    model.eval()
    criterion_mse_t = nn.MSELoss().to(device)
    criterion_bce_pressure_t = nn.BCELoss().to(device)

    epoch_loss_t = []
    for i, (en_in_t, en_time_t, de_in_t, de_time_t, _) in enumerate(data_loader): 
        en_in_t = en_in_t.float().to(device)
        en_time_t = en_time_t.float().to(device)
        
        # true value for forecasting
        true_t = de_in_t[:,args.label_len:,:].float().to(device) 

        # zero padding for decoder input (for forecasting part)
        dec_inp_t = torch.zeros([de_in_t.shape[0], args.pred_len, de_in.shape[-1]]).float()

        de_in_t = torch.cat([de_in_t[:,:args.label_len,:], dec_inp_t], dim=1).float().to(device)
        de_time_t = de_time_t.float().to(device)

        _, forecasting_output_t = model(en_in_t, en_time_t, de_in_t, de_time_t)

        if args.use_feat == "all":
            forecast_traj_right_t, true_traj_right_t = forecasting_output_t[:, :, 0:3], true_t[:, :, 0:3]
            forecast_quat_right_t, true_quat_right_t = forecasting_output_t[:, :, 3:7], true_t[:, :, 3:7]
            forecast_pres_right_t, true_pres_right_t = forecasting_output_t[:, :, 7], true_t[:, :, 7]

            forecast_traj_head_t, true_traj_head_t = forecasting_output_t[:, :, 8:11], true_t[:, :, 8:11]
            forecast_quat_head_t, true_quat_head_t = forecasting_output_t[:, :, 11:15], true_t[:, :, 11:15]

            forecast_traj_left_t, true_traj_left_t = forecasting_output_t[:, :, 15:18], true_t[:, :, 15:18]
            forecast_quat_left_t, true_quat_left_t = forecasting_output_t[:, :, 18:22], true_t[:, :, 18:22]
            forecast_pres_left_t, true_pres_left_t = forecasting_output_t[:, :, 22], true_t[:, :, 22]

            forecast_traj_quat_t = torch.cat([forecast_traj_right_t, forecast_quat_right_t, forecast_traj_head_t, forecast_quat_head_t, forecast_traj_left_t, forecast_quat_left_t], dim=2)
            true_traj_quat_t = torch.cat([true_traj_right_t, true_quat_right_t, true_traj_head_t, true_quat_head_t, true_traj_left_t, true_quat_left_t], dim=2)

            mse_loss_t = criterion_mse_t(forecast_traj_quat_t, true_traj_quat_t)    # trajectory+orientation loss

            m_t = nn.Sigmoid()
            pred_pressure_t = torch.flatten(torch.cat([forecast_pres_right_t, forecast_pres_left_t], dim=1))
            pred_pressure_t = m_t(pred_pressure_t)
            true_pressure_t = torch.flatten(torch.cat([true_pres_right_t, true_pres_left_t], dim=1))
            condition_t = true_pressure_t <= 0.5
            true_pressure_t = torch.where(condition_t, torch.tensor(0., dtype=torch.float32).to(device), torch.tensor(1., dtype=torch.float32).to(device))
            bce_pressure_loss_t = criterion_bce_pressure_t(pred_pressure_t, true_pressure_t)
            # print(bce_pressure_loss)
        else:
            print("use_feat should be all at this moment")
            exit()
        loss_t = 0.95 * mse_loss_t + 0.05 * bce_pressure_loss_t
        epoch_loss_t.append(loss_t.item())

    checkpoint_path = os.path.join(args.ckp_out_root, args.train_for, f"{args.use_feat}_checkpoints")
    if np.mean(epoch_loss_t) < global_loss:
        global_loss = np.mean(epoch_loss_t)
        # best_epoch_acc = epoch + 1

        torch.save(
            model.state_dict(),
            os.path.join(
                checkpoint_path,
                "finetune_test_{}_{}_{}_sl{}_ll{}_pl{}.pth".format(
                    args.train_for,
                    test_data_name,
                    args.use_feat,
                    str(args.seq_len).zfill(2),
                    str(args.label_len).zfill(2),
                    str(args.pred_len).zfill(2),
                )                            
            )
        )

    
    return np.mean(epoch_loss_t), global_loss


def get_argumets():
    """
        Parse arguments from command line
    """
    parser = argparse.ArgumentParser(description='cross system VR auth')
    parser.add_argument('--data_root', type=str, required=False, default="splited_data",help='')
    parser.add_argument('--out_root', type=str, required=False, default='finetune',help='')
    parser.add_argument('--ckp_in_root', type=str, required=False, default='forecast_only/Data_1/all_checkpoints',help='')
    parser.add_argument('--ckp_out_root', type=str, required=False, default='finetune',help='')
    parser.add_argument('--data', type=str, required=False, default='Data_2',help='')
    parser.add_argument('--train_for', type=str, required=False, default='Quest1',help='')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--learning_rate', type=int, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--max_epoch', type=int, default=125, help='')

    parser.add_argument('--seq_len', type=int, default=25, help='input sequence length of Informer encoder')
    parser.add_argument('--label_len', type=int, default=10, help='start token length of Informer decoder')
    parser.add_argument('--pred_len', type=int, default=40, help='prediction sequence length')
    parser.add_argument("--use_feat", type=str, default="all", choices=["all", "right_all", "right_traj"], help='')

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

    logger = logging.getLogger("V3 finetune_{}_sl{}_ll{}_pl{}".format(
        args.train_for,
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
        log_out_to = os.path.join(args.out_root, args.train_for, f"{args.use_feat}_log")
        if not os.path.exists(log_out_to):
            os.makedirs(log_out_to)

        fh = logging.FileHandler(os.path.join(
            log_out_to,
            "finetune_train_{}_{}_sl{}_ll{}_pl{}.log".format(
                args.train_for,
                args.use_feat,
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

    logger.debug("Features:             {}".format(args.use_feat))
    logger.debug("Data:                 {}".format(args.data))
    logger.debug("Train for:            {}".format(args.train_for))
    logger.debug("Sequence length:      {}".format(args.seq_len))
    logger.debug("Label length:         {}".format(args.label_len))
    logger.debug("Prediction length:    {}".format(args.pred_len))
    logger.debug("Number of epoch:      {}".format(args.max_epoch))
    logger.debug("Batch size:           {}".format(args.batch_size))
    logger.debug("Learning rate:        {}".format(args.learning_rate))
    logger.debug("Dropout rate:         {}".format(args.dropout))

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    logger.info("Using device:          {}\n".format(device))


    # Dataloader for training
    V3_train = Forecast_FineTune(
        root = args.data_root, 
        use_data=args.data,
        train_for=args.train_for,          
        forecasting_sizes=[args.seq_len, args.label_len, args.pred_len],
        use_feat = args.use_feat
    )
    V3_train_loader = DataLoader(
        V3_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False
    )

    # Dataloader for other dataset (for validation)
    # Q1 -- Q2, global_val_loss_1
    Q2_test = Forecast_FineTune(
        root = args.data_root, 
        use_data=args.data,
        train_for="Quest2",          
        forecasting_sizes=[args.seq_len, args.label_len, args.pred_len],
        use_feat = args.use_feat
    )
    Q2_test_loader = DataLoader(
        Q2_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True
    )

    # Q1 -- V1, global_val_loss_2
    V1_test = Forecast_FineTune(
        root = args.data_root, 
        use_data=args.data,
        train_for="Vive1",          
        forecasting_sizes=[args.seq_len, args.label_len, args.pred_len],
        use_feat = args.use_feat
    )
    V1_test_loader = DataLoader(
        V1_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True
    )

    # Q1 -- V2, global_val_loss_3
    V2_test = Forecast_FineTune(
        root = args.data_root, 
        use_data=args.data,
        train_for="Vive2",          
        forecasting_sizes=[args.seq_len, args.label_len, args.pred_len],
        use_feat = args.use_feat
    )
    V2_test_loader = DataLoader(
        V2_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True
    )

    # Q1 -- C1, global_val_loss_4
    C1_test = Forecast_FineTune(
        root = args.data_root, 
        use_data=args.data,
        train_for="Cosmos1",          
        forecasting_sizes=[args.seq_len, args.label_len, args.pred_len],
        use_feat = args.use_feat
    )
    C1_test_loader = DataLoader(
        C1_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True
    )

    # Q1 -- C2, global_val_loss_5
    C2_test = Forecast_FineTune(
        root = args.data_root, 
        use_data=args.data,
        train_for="Cosmos2",          
        forecasting_sizes=[args.seq_len, args.label_len, args.pred_len],
        use_feat = args.use_feat
    )
    C2_test_loader = DataLoader(
        C2_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True
    )

    # load pretrained model
    ckp_path = os.path.join(
       args.ckp_in_root, 
       "forecasting_only_train_all_Data_1_sl{}_ll{}_pl{}.pth".format(
            str(args.seq_len).zfill(2),
            str(args.label_len).zfill(2),
            str(args.pred_len).zfill(2)
       )
    )

    # build model
    model = V3_Forecast_Model(
        enc_in=num_feat, dec_in=num_feat, c_out=num_feat, 
        seq_len=args.seq_len, label_len=args.label_len, out_len=args.pred_len,
        num_feats=num_feat, d_model=args.d_model, n_heads=args.n_heads,
        e_layers=args.e_layers, d_layers=args.d_layers, 
        d_ff=args.d_ff, dropout=args.dropout, activation='gelu', output_attention=False,
        device=device
    ).to(device)

    model.load_state_dict(torch.load(ckp_path, map_location=device))

    # optimizer & loss functions
    model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion_mse =  nn.MSELoss().to(device)
    criterion_bce_pressure =  nn.BCELoss().to(device)

    # checkpoints out
    checkpoint_path = os.path.join(args.ckp_out_root, args.train_for, f"{args.use_feat}_checkpoints")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    global_train_loss = np.inf
    global_val_loss_1 = np.inf
    global_val_loss_2 = np.inf
    global_val_loss_3 = np.inf
    global_val_loss_4 = np.inf
    global_val_loss_5 = np.inf
    for epoch in tqdm.tqdm(range(args.max_epoch)):
        model.train()
        epoch_loss = 0

        ### train ###
        for i, (en_in, en_time, de_in, de_time, _) in enumerate(V3_train_loader):
            model_optim.zero_grad()

            en_in = en_in.float().to(device)
            en_time = en_time.float().to(device)

            # true value for forecasting
            true = de_in[:,args.label_len:,:].float().to(device)

            # zero padding for decoder input (for forecasting part)
            dec_inp = torch.zeros([de_in.shape[0], args.pred_len, de_in.shape[-1]]).float()
            de_in = torch.cat([de_in[:,:args.label_len,:], dec_inp], dim=1).float().to(device)
            de_time = de_time.float().to(device)

            # forecasting
            _, forecasting_output = model(en_in, en_time, de_in, de_time)
            
            if args.use_feat == "all":
                forecast_traj_right, true_traj_right = forecasting_output[:, :, 0:3], true[:, :, 0:3]
                forecast_quat_right, true_quat_right = forecasting_output[:, :, 3:7], true[:, :, 3:7]
                forecast_pres_right, true_pres_right = forecasting_output[:, :, 7], true[:, :, 7]

                forecast_traj_head, true_traj_head = forecasting_output[:, :, 8:11], true[:, :, 8:11]
                forecast_quat_head, true_quat_head = forecasting_output[:, :, 11:15], true[:, :, 11:15]

                forecast_traj_left, true_traj_left = forecasting_output[:, :, 15:18], true[:, :, 15:18]
                forecast_quat_left, true_quat_left = forecasting_output[:, :, 18:22], true[:, :, 18:22]
                forecast_pres_left, true_pres_left = forecasting_output[:, :, 22], true[:, :, 22]

                forecast_traj_quat = torch.cat([forecast_traj_right, forecast_quat_right, forecast_traj_head, forecast_quat_head, forecast_traj_left, forecast_quat_left], dim=2)
                true_traj_quat = torch.cat([true_traj_right, true_quat_right, true_traj_head, true_quat_head, true_traj_left, true_quat_left], dim=2)

                mse_loss = criterion_mse(forecast_traj_quat, true_traj_quat)    # trajectory+orientation loss

                m = nn.Sigmoid()
                pred_pressure = torch.flatten(torch.cat([forecast_pres_right, forecast_pres_left], dim=1))
                pred_pressure = m(pred_pressure)
                true_pressure = torch.flatten(torch.cat([true_pres_right, true_pres_left], dim=1))
                condition = true_pressure <= 0.5
                true_pressure = torch.where(condition, torch.tensor(0., dtype=torch.float32).to(device), torch.tensor(1., dtype=torch.float32).to(device))
                bce_pressure_loss = criterion_bce_pressure(pred_pressure, true_pressure)
            else:
                print("use_feat should be all at this moment")
                exit()

            loss = 0.95 * mse_loss + 0.05 * bce_pressure_loss

            loss.backward()
            model_optim.step()
            epoch_loss += loss.item()

        logger.info("Epoch {}/{} tra+orien loss: {} | pressure loss: {} | total loss: {}".format(
            epoch + 1, 
            args.max_epoch,
            mse_loss.item(),
            bce_pressure_loss.item(),
            epoch_loss / len(V3_train_loader)
        ))

        # save the model that has the smallest training loss
        if epoch_loss < global_train_loss:
            global_train_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_path, "finetune_train_{}_{}_sl{}_ll{}_pl{}.pth".format(
                args.train_for,
                args.use_feat,
                str(args.seq_len).zfill(2),
                str(args.label_len).zfill(2),
                str(args.pred_len).zfill(2)
            )))

        ### validation ###
        with torch.no_grad():
            if args.train_for == "Quest1":
                test_loss_mean_1, global_val_loss_1 = validation(
                        model=model,
                        test_data_name="Quest2",
                        data_loader=Q2_test_loader,
                        global_loss=global_val_loss_1,
                        device=device,
                        args=args,
                    )
                logger.info("Testing with {} in epoch {}/{} with testing loss {}.".format("Quest 2", epoch+1, args.max_epoch, test_loss_mean_1))
                
                test_loss_mean_2, global_val_loss_2 = validation(
                        model=model,
                        test_data_name="Vive1",
                        data_loader=V1_test_loader,
                        global_loss=global_val_loss_2,
                        device=device,
                        args=args,
                    )
                logger.info("Testing with {} in epoch {}/{} with testing loss {}.".format("Vive 1", epoch+1, args.max_epoch, test_loss_mean_2))

                test_loss_mean_3, global_val_loss_3 = validation(
                        model=model,
                        test_data_name="Vive2",
                        data_loader=V2_test_loader,
                        global_loss=global_val_loss_3,
                        device=device,
                        args=args,
                    )
                logger.info("Testing with {} in epoch {}/{} with testing loss {}.".format("Vive 2", epoch+1, args.max_epoch, test_loss_mean_3))

                test_loss_mean_4, global_val_loss_4 = validation(
                        model=model,
                        test_data_name="Cosmos1",
                        data_loader=C1_test_loader,
                        global_loss=global_val_loss_4,
                        device=device,
                        args=args,
                    )
                logger.info("Testing with {} in epoch {}/{} with testing loss {}.".format("Cosmos 1", epoch+1, args.max_epoch, test_loss_mean_4))

                test_loss_mean_5, global_val_loss_5 = validation(
                        model=model,
                        test_data_name="Cosmos2",
                        data_loader=C2_test_loader,
                        global_loss=global_val_loss_5,
                        device=device,
                        args=args,
                    )
                logger.info("Testing with {} in epoch {}/{} with testing loss {}.".format("Cosmos 2", epoch+1, args.max_epoch, test_loss_mean_5))
            
            elif args.train_for == "Quest2":           
                test_loss_mean_2, global_val_loss_2 = validation(
                        model=model,
                        test_data_name="Vive1",
                        data_loader=V1_test_loader,
                        global_loss=global_val_loss_2,
                        device=device,
                        args=args,
                    )
                logger.info("Testing with {} in epoch {}/{} with testing loss {}.".format("Vive 1", epoch+1, args.max_epoch, test_loss_mean_2))

                test_loss_mean_3, global_val_loss_3 = validation(
                        model=model,
                        test_data_name="Vive2",
                        data_loader=V2_test_loader,
                        global_loss=global_val_loss_3,
                        device=device,
                        args=args,
                    )
                logger.info("Testing with {} in epoch {}/{} with testing loss {}.".format("Vive 2", epoch+1, args.max_epoch, test_loss_mean_3))

                test_loss_mean_4, global_val_loss_4 = validation(
                        model=model,
                        test_data_name="Cosmos1",
                        data_loader=C1_test_loader,
                        global_loss=global_val_loss_4,
                        device=device,
                        args=args,
                    )
                logger.info("Testing with {} in epoch {}/{} with testing loss {}.".format("Cosmos 1", epoch+1, args.max_epoch, test_loss_mean_4))

                test_loss_mean_5, global_val_loss_5 = validation(
                        model=model,
                        test_data_name="Cosmos2",
                        data_loader=C2_test_loader,
                        global_loss=global_val_loss_5,
                        device=device,
                        args=args,
                    )
                logger.info("Testing with {} in epoch {}/{} with testing loss {}.".format("Cosmos 2", epoch+1, args.max_epoch, test_loss_mean_5))
            elif args.train_for == "Vive1":           
                test_loss_mean_3, global_val_loss_3 = validation(
                        model=model,
                        test_data_name="Vive2",
                        data_loader=V2_test_loader,
                        global_loss=global_val_loss_3,
                        device=device,
                        args=args,
                    )
                logger.info("Testing with {} in epoch {}/{} with testing loss {}.".format("Vive 2", epoch+1, args.max_epoch, test_loss_mean_3))

                test_loss_mean_4, global_val_loss_4 = validation(
                        model=model,
                        test_data_name="Cosmos1",
                        data_loader=C1_test_loader,
                        global_loss=global_val_loss_4,
                        device=device,
                        args=args,
                    )
                logger.info("Testing with {} in epoch {}/{} with testing loss {}.".format("Cosmos 1", epoch+1, args.max_epoch, test_loss_mean_4))

                test_loss_mean_5, global_val_loss_5 = validation(
                        model=model,
                        test_data_name="Cosmos2",
                        data_loader=C2_test_loader,
                        global_loss=global_val_loss_5,
                        device=device,
                        args=args,
                    )
                logger.info("Testing with {} in epoch {}/{} with testing loss {}.".format("Cosmos 2", epoch+1, args.max_epoch, test_loss_mean_5))
            elif args.train_for == "Vive2":           
                test_loss_mean_4, global_val_loss_4 = validation(
                        model=model,
                        test_data_name="Cosmos1",
                        data_loader=C1_test_loader,
                        global_loss=global_val_loss_4,
                        device=device,
                        args=args,
                    )
                logger.info("Testing with {} in epoch {}/{} with testing loss {}.".format("Cosmos 1", epoch+1, args.max_epoch, test_loss_mean_4))

                test_loss_mean_5, global_val_loss_5 = validation(
                        model=model,
                        test_data_name="Cosmos2",
                        data_loader=C2_test_loader,
                        global_loss=global_val_loss_5,
                        device=device,
                        args=args,
                    )
                logger.info("Testing with {} in epoch {}/{} with testing loss {}.".format("Cosmos 2", epoch+1, args.max_epoch, test_loss_mean_5))
            elif args.train_for == "Cosmos1":           
                test_loss_mean_5, global_val_loss_5 = validation(
                        model=model,
                        test_data_name="Cosmos2",
                        data_loader=C2_test_loader,
                        global_loss=global_val_loss_5,
                        device=device,
                        args=args,
                    )
                logger.info("Testing with {} in epoch {}/{} with testing loss {}.".format("Cosmos 2", epoch+1, args.max_epoch, test_loss_mean_5))
            else:
                print("Wrong training data name!")
                exit()

    logger.info("Training finished.")
    torch.cuda.empty_cache()