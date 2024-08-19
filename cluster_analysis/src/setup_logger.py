import logging
root_path = '/home/ddeandres'


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(f'{root_path}/experiment.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('UNSW')
logger.info(f'Logging to {root_path}/experiment.log')