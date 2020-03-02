import pandas as pd


def extract_date(dataframe, date_feature='datetime'):
    df_ = dataframe.copy()
    time_data = pd.DatetimeIndex(df_[date_feature])
    df_['date'] = time_data.date
    df_['day'] = time_data.day
    df_['month'] = time_data.month
    df_['year'] = time_data.year
    df_['hour'] = time_data.hour
    df_['day_of_week'] = time_data.dayofweek
    df_['week_of_year'] = time_data.weekofyear


def train_test_split_by_day(df, cutoff_day: int, day_feature='day'):
    train = df[df[day_feature] <= cutoff_day].reset_index(drop=True)
    test = df[df[day_feature] > cutoff_day].reset_index(drop=True)
    return train, test
