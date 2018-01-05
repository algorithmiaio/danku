import config as cfg

def dprint(message):
    if cfg.DEBUG:
        print(message + "\n")
    else:
        pass
