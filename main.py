import os
import configparser
from src.tcn_3cd import C3DTCN
from src.class_net import ClassNet

def main():

    config = configparser.ConfigParser()
    path = os.path.abspath("./")
    config.read(os.path.join(path, "config.ini"))

    verbosity = config.getboolean("GLOBAL", "verbose")
    tcn_settings = config["TCN"]
    cnet_settings = config["CNET"]

    tcn = C3DTCN(tcn_settings, verbose=verbosity)
    cnet = ClassNet(cnet_settings, tcn,  verbose=verbosity)


if __name__ == '__main__':
    main()
