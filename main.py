import os
import configparser
from src.tcn_3cd import C3DTCN

def main():

    config = configparser.ConfigParser()
    path = os.path.abspath("./")
    config.read(os.path.join(path, "config.ini"))

    verbose = config.getboolean("GLOBAL", "verbose")

    tcn = get_tcn(config, verbose)

def get_tcn(config, verbose):
    tcn_settings = config["TCN"]
    train = tcn_settings.getboolean("train")
    if train or not os.path.exists("./saves/tcnc3d.pt"):
        tcn = C3DTCN.train_tcn(tcn_settings, verbose)
    else:
        tcn = C3DTCN.load_tcn(tcn_settings)

    return tcn

if __name__ == '__main__':
    main()
