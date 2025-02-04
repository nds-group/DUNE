import logging
root_path = '/home/beyzabutun/distributed_in_band/model_sequencing/src'


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

logger = logging.getLogger('TON-IOT')
logger.info(f'Logging to {root_path}/experiment.log')