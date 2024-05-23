import configparser
import os

path = os.getenv('CONFIG_PATH',"./config.ini")
print(r"path")

config = configparser.ConfigParser()
config.read(path)
param_config = config["parameters"]

