



def datestamp() -> str:
    from datetime import datetime
    return datetime.today().strftime("%y%m%d")