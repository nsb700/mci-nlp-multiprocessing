BATCH_SIZE = 100
THRESHOLD = 0.6
PROCESS_COUNT = 10
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "format": "%(name)s %(asctime)s %(levelname)s %(filename)s %(lineno)s %(process)d %(message)s",
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
        }
    },
    "handlers": {
        "runlogfile": {
            "class": "logging.FileHandler",
            "filename": "02_finalresult/runlog.txt",
            "formatter": "json",
            "level": "INFO",
        }
    },
    "loggers": {
        "": {
            "handlers": ["runlogfile"], 
            "level": "DEBUG"
        }
    },
}