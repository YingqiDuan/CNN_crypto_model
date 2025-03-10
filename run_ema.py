import schedule
import time
from ema import main


# pacific time

schedule.every().day.at("15:50").do(lambda: main())


while True:
    schedule.run_pending()
    time.sleep(1)
