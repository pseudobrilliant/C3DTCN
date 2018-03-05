import os
import configparser
from src.tcn_3cd import C3DTCN
from src.class_net import ClassNet
from src.ixmas_dataset import IXMASDataset
from torchvision import transforms

def main():

    config = configparser.ConfigParser()
    path = os.path.abspath("./")
    config.read(os.path.join(path, "config.ini"))

    verbosity = config.getboolean("GLOBAL", "verbose")
    tcn_settings = config["TCN"]
    cnet_settings = config["CNET"]

    tcn = C3DTCN(tcn_settings, verbose=verbosity)
    cnet = ClassNet(cnet_settings, tcn,  verbose=verbosity)

    testing_collections = cnet_settings["testing"].split(',')
    transform = transforms.Compose([transforms.CenterCrop(224), transforms.Resize(112), transforms.ToTensor()])
    path = os.path.abspath("./")
    training = IXMASDataset(path, testing_collections, transform=transform, verbose=False)
    training.set_triplet_flag(False)
    cnet.test(testing_collections)

if __name__ == '__main__':
    main()
