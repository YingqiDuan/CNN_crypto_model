import schedule
import time


def job():
    print("Running the script...")
    import predict

    predict.main()


# 定义每天运行时间
schedule.every().day.at("04:30").do(job)
schedule.every().day.at("08:30").do(job)
schedule.every().day.at("12:30").do(job)
schedule.every().day.at("16:30").do(job)
schedule.every().day.at("20:30").do(job)
schedule.every().day.at("24:30").do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
