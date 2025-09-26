import logging

def get_logger(name: str):
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)s | %(message)s')
    ch.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.addHandler(ch)
    return logger
