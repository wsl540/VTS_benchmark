from torch.utils.data import DataLoader
from data_provider.data_loader import UCRloader,DWTloader
from data_provider.ucr import collate_fn

data_dict={
    'VARY_UCR':UCRloader,
    'DWT':DWTloader,
}

def data_provider(args,flag):
    Data=data_dict[args.data]
    if flag=='TEST':
        shuffle_flag=False
        batch_size=args.batch_size
    else:
        shuffle_flag=True
        batch_size=args.batch_size
    data_set=Data(
        args=args,
        flag=flag,
    )


    data_loader=DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=lambda x: collate_fn(x, args)
    )
    return data_set,data_loader


