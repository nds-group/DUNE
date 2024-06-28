import logging
root_path = '/home/ddeandres/distributed_in_band/UNSW/cluster_model_analysis_results/correlation_analysis'


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(f'{root_path}/experiment.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('UNSW')