import argparse

## Arg parsing
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="gtsrb", help="The name of the dataset")
    parser.add_argument("--model_name", type=str, default="resnext50_32x4d", help="The name of the model")
    parser.add_argument("--batch_size", type=int, default=1, help="")
    parser.add_argument("--device", type=str, default="cuda:0", help="Options are are cpu or cuda:0")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--report_location", type=str, default="./report.md", help="The location where the generated report will be stored")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of dataloader workers")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--dataloaderv", type=int, default=1)

    args = parser.parse_args()
    dataset = args.dataset
    model_name = args.model_name
    batch_size = args.batch_size
    device = args.device
    num_epochs = args.num_epochs
    report_location = args.report_location
    num_workers = args.num_workers
    shuffle = args.shuffle
    dataloaderv = args.dataloaderv
    return dataset,model_name,batch_size,device,num_epochs,num_workers,shuffle,dataloaderv