import logging
from src.utils.data_processing import read_rte_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_file(path: str):
    for year in range(2018, 2024):
        logging.info(f"File {year}")
        if year==2020:
            continue
        df = read_rte_file(path + f"eCO2mix_RTE_Annuel-Definitif_{year}.xls")
        df.to_csv(path + f'{year}.csv', index=False)

if __name__ == "__main__":
    logging.info("Process raw rte data file to csv")
    path = "data/load_rte/"
    process_file(path)
