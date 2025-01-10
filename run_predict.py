import schedule
import time


def job():
    print("Running the script...")
    import predict

    predict.main()


# 定义每天运行时间
schedule.every().day.at("03:30").do(job)
schedule.every().day.at("07:30").do(job)
schedule.every().day.at("11:30").do(job)
schedule.every().day.at("15:30").do(job)
schedule.every().day.at("19:30").do(job)
schedule.every().day.at("23:30").do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
