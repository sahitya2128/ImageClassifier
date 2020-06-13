import loadTheData
import build_train_model

from getInputArgs import getInputArgs


def main():
    
    in_arg = getInputArgs()
    
    data_dir = in_arg.data_dir
    save_dir = in_arg.save_dir
    arch = in_arg.arch
    lr = in_arg.lr
    hidden_units = in_arg.hidden_units
    epochs = in_arg.epochs
    gpu = in_arg.gpu
    
    train_dataloaders, vaild_dataloaders, test_dataloaders, class_to_idx = loadTheData.loadData(data_dir)
    model, criterion, optimizer = build_train_model.setup_model(structure = arch, dropout = 0.5, lr=lr, power = gpu, hidden_layer = hidden_units)
    build_train_model.train_model(model, criterion, optimizer, train_dataloaders, vaild_dataloaders, power = gpu, epochs = epochs)
    build_train_model.save_model(class_to_idx, save_dir, model, arch, optimizer)
    
    
if __name__ == "__main__":
    main()