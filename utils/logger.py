import os
import sys
from datetime import datetime

def activer_log(dossier_log):
    os.makedirs(dossier_log, exist_ok=True)
    now = datetime.now().strftime('%Y_%m_%d_%H_%M')
    log_filename = os.path.join(dossier_log, f"log_{now}.txt")

    class Logger:
        def __init__(self, stream, filepath):
            self.terminal = stream
            self.log = open(filepath, "a", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(sys.stdout, log_filename)
    return sys.stdout
