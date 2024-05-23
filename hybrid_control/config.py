import configparser

config = configparser.ConfigParser()
config.read("./config.ini")
param_config = config["parameters"]
