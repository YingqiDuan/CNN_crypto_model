import schedule
import time


def job():
    print("Running the script...")
    import predict

    predict.main()


# 定义每天运行时间
schedule.every().day.at("03:50").do(job)
schedule.every().day.at("07:50").do(job)
schedule.every().day.at("11:50").do(job)
schedule.every().day.at("15:50").do(job)
schedule.every().day.at("19:50").do(job)
schedule.every().day.at("23:50").do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
