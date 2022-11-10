import configparser
from pathlib import Path
from loguru import logger


BASE_PATH = Path(__file__).parent.parent.absolute()
CONFIG_PATH = BASE_PATH / "config.ini"

config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read(CONFIG_PATH)

logger.info("config['Paths']={}".format(list(config["Paths"].items())))
