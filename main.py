from download_data import download_data, get_user_input
from merge_data import merge_data


def main():
    days, trading_pairs, periods = get_user_input()
    download_data(days, trading_pairs, periods)
    merge_data(trading_pairs, periods)


if __name__ == "__main__":
    main()
