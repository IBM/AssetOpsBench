import logging

# Function to get the custom logger
def get_custom_logger(name):
    # Create a logger object
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set the minimum log level
    
    allowed_logger = [__name__, 'reactxen', 'cbm_gen', 'agent_hive']
    for log_name, log_obj in logging.Logger.manager.loggerDict.items():
        if log_name not in allowed_logger:
            logging.getLogger(log_name).setLevel(logging.INFO)

    logging.getLogger("ibm_watsonx_ai").disabled = True
    logging.getLogger("httpx").disabled = True

    # Check if handlers are already added to avoid duplicate logs
    if not logger.handlers:
        # Create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Add formatter to ch
        ch.setFormatter(formatter)

        # Add ch to logger
        logger.addHandler(ch)
    
    return logger