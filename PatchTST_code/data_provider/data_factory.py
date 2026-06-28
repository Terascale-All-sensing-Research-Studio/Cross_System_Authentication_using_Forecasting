from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Forecast_Only_Dataset
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'general_forecast': Forecast_Only_Dataset,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    if args.data == "general_forecast":
        data_set = Data(
            root="splited_data",
            use_data="Data_1",
            split="train",
            forecasting_sizes=[args.seq_len, args.label_len, args.pred_len],
        )
    else:
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq
        )
        print(flag, len(data_set))


    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True)

    return data_set, data_loader
