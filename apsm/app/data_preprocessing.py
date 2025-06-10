import pandas as pd
import numpy as np
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer, TargetEncoder


RANDOM_STATE = 23


def preprocess_time_series(
    df: pd.DataFrame,
    target: str,
    transformers: dict = None,
    do_scale: bool = True,
    do_diff: bool = True,
    do_yeo_johnson: bool = True,
    do_window_normalizing: bool = True,
    window_size: int = 10,
    do_encode: bool = True,
    is_train: bool = False
):
    """
    Предобрабатывает временной ряд для столбца 'target' в DataFrame, изменяя только этот столбец.
    Возвращает преобразованный DataFrame с индексом 'Date' и словарь трансформеров для обратного преобразования.

    Parameters
    ----------
    df : pd.DataFrame
        Исходный DataFrame.
    target : str
        Название столбца, для которого выполняются преобразования.
    transformers : dict, optional
        Словарь с объектами трансформеров (используется при is_train=False).
    do_scale : bool, по умолчанию True
        Выполнять ли масштабирование.
    do_diff : bool, по умолчанию True
        Выполнять ли дифференцирование.
    do_yeo_johnson : bool, по умолчанию True
        Выполнять ли преобразование Yeo-Johnson.
    do_window_normalizing : bool, по умолчанию True
        Выполнять ли оконную нормализацию.
    window_size : int, по умолчанию 10
        Размер окна для оконной нормализации.
    do_encode : bool
        Выполнять ли кодировку категориальных признаков.
    is_train : bool, по умолчанию False
        Флаг, указывающий, являются ли данные тренировочными.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        Преобразованный DataFrame (индекс 'Date') и словарь трансформеров для восстановления значений.
    """
    df_transformed = df.copy()
    ts = df_transformed[target].copy()

    if is_train:
        transformers = {}
    elif transformers is None:
        raise ValueError('Для тестовых данных необходимо передать transformers.')

    # Масштабирование.
    if do_scale:
        values = ts.values.reshape(-1, 1)

        if is_train:
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(values)
            transformers['scaler'] = scaler
        else:
            scaler = transformers['scaler']
            scaled_values = scaler.transform(values)

        ts = pd.Series(
            data=scaled_values.flatten(),
            index=ts.index,
            name=ts.name
        )

    # Дифференцирование.
    if do_diff:
        transformers['last_value'] = ts.iloc[0]
        ts = ts.diff()
        ts.dropna(inplace=True)

    # Преобразование Yeo-Johnson.
    if do_yeo_johnson:
        if is_train:
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            non_nan_values = ts.values[~np.isnan(ts.values)].reshape(-1, 1)
            if len(non_nan_values) > 0:
                pt.fit(non_nan_values)
            transformers['yeo_johnson'] = pt

        pt = transformers['yeo_johnson']
        transformed_values = np.full_like(ts.values, np.nan, dtype=float)
        non_nan_mask = ~np.isnan(ts.values)
        non_nan_values = ts.values[non_nan_mask].reshape(-1, 1)

        if len(non_nan_values) > 0:
            transformed_values[non_nan_mask] = pt.transform(non_nan_values).flatten()

        ts = pd.Series(transformed_values, index=ts.index, name=ts.name)

    # Оконная нормализация.
    if do_window_normalizing:
        ts_rolled = ts.rolling(window=window_size)
        rolled_mean = ts_rolled.mean()
        rolled_std = ts_rolled.std()
        transformers['rolled_mean'] = rolled_mean
        transformers['rolled_std'] = rolled_std
        transformers['window_init_values'] = ts.iloc[:window_size - 1].values.copy()

        eps = 1e-9
        ts = (ts - rolled_mean) / (rolled_std + eps)
        ts = ts[window_size - 1:]

    df_transformed[target] = ts
    df_transformed.dropna(inplace=True)

    if do_encode:
        if is_train:
            encoder = TargetEncoder(cv=3, random_state=RANDOM_STATE)
            df_transformed['ticker'] = encoder.fit_transform(X=df_transformed[['ticker']], y=ts)

            transformers['encoder'] = encoder
        else:
            encoder = transformers['encoder']
            df_transformed['ticker'] = encoder.transform(X=df_transformed[['ticker']])

    return df_transformed.set_index('Date'), transformers


def inverse_preprocess_time_series(
    ts_transformed: np.array,
    transformers: dict,
    do_scale: bool = True,
    do_diff: bool = True,
    do_yeo_johnson: bool = True,
    do_window_normalizing: bool = True,
    window_size: int = 10
) -> np.array:
    """
    Восстанавливает исходный временной ряд из преобразованного np.array,
    используя сохранённые параметры трансформаций, выполненных в функции preprocess_time_series.

    Parameters
    ----------
    ts_transformed : np.array
        Преобразованный временной ряд.
    transformers : dict
        Словарь с сохранёнными объектами и параметрами трансформаций.
    do_scale : bool, по умолчанию True
        Выполнять ли масштабирование.
    do_diff : bool, по умолчанию True
        Выполнять ли дифференцирование.
    do_yeo_johnson : bool, по умолчанию True
        Выполнять ли преобразование Yeo-Johnson.
    do_window_normalizing : bool, по умолчанию True
        Выполнять ли оконную нормализацию.
    window_size : int, по умолчанию 10
        Размер окна для оконной нормализации.

    Returns
    -------
    np.array
        Восстановленный временной ряд в исходных единицах измерения.
    """
    ts = ts_transformed.copy()

    # Обратная оконная нормализация.
    if do_window_normalizing:
        eps = 1e-9
        rolled_mean = transformers.get('rolled_mean')
        rolled_std = transformers.get('rolled_std')
        init_values = transformers.get('window_init_values', np.array([]))

        if rolled_mean is None or rolled_std is None:
            raise ValueError("Нет данных для обратной оконной нормализации.")

        ts = ts * (rolled_std.values[window_size - 1:window_size - 1 + len(ts)] + eps)\
                 + rolled_mean.values[window_size - 1:window_size - 1 + len(ts)]
        ts = np.concatenate([init_values, ts])

    # Обратное Yeo-Johnson.
    if do_yeo_johnson and 'yeo_johnson' in transformers:
        pt = transformers['yeo_johnson']
        ts = pt.inverse_transform(ts.reshape(-1, 1)).flatten()

    # Обратное дифференцирование.
    if do_diff and 'last_value' in transformers:
        ts = np.cumsum(ts, dtype=np.float64) + transformers['last_value']

    # Обратное масштабирование.
    if do_scale and 'scaler' in transformers:
        scaler = transformers['scaler']
        ts = scaler.inverse_transform(ts.reshape(-1, 1)).flatten()

    return ts[window_size - 1:]


def extract_time_series_features(
    df: pd.Series,
    lags=[1, 2, 3, 4, 5],
    rolling_windows=[7, 30],
    fourier_windows=[30]
):
    """
    Извлекает расширенные признаки из временного ряда. Требует, чтобы дата была в колонке 'Date'.

    Parameters
    ----------
    df : pd.Series или pd.DataFrame
        Временной ряд (с колонкой 'Date').
    lags : list of int, по умолчанию [1, 2, 3, 4, 5]
        Лаги для расчета лаговых и разностных признаков.
    rolling_windows : list of int, по умолчанию [7, 30]
        Окна для расчета скользящих статистик (mean, std, min, max, медиана, квантили).
    fourier_windows : list of int, по умолчанию [30]
        Окна для расчета признаков на основе FFT (доминирующая частота).

    Returns
    -------
    pd.DataFrame
        DataFrame с извлечёнными признаками. Требует, чтобы дата была в колонке 'Date'.
    """
    if isinstance(df, pd.Series):
        df = df.to_frame(name='value')
    else:
        df = df.copy()
        if 'value' not in df.columns:
            df = df.iloc[:, [0]]
            df.columns = ['value']

    df = df.sort_values('Date').reset_index(drop=True)
    df['weekday'] = df['Date'].dt.dayofweek

    # Календарные признаки.
    df['day_of_week'] = df['weekday']
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Лаговые признаки.
    for lag in lags:
        df[f'lag_{lag}'] = df['value'].shift(lag)

    # Разностные признаки.
    for lag in lags:
        df[f'diff_{lag}'] = df['value'] - df['value'].shift(lag)

    # Скользящие статистики (обычные).
    for window in rolling_windows:
        df[f'rolling_mean_{window}'] = df['value'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['value'].rolling(window=window).std()
        df[f'rolling_min_{window}'] = df['value'].rolling(window=window).min()
        df[f'rolling_max_{window}'] = df['value'].rolling(window=window).max()

    # FFT-признаки.
    def compute_dominant_freq(arr):
        if np.all(np.isnan(arr)):
            return np.nan

        arr_filled = pd.Series(arr).ffill().values
        fft_vals = fft(arr_filled)
        power = np.abs(fft_vals) ** 2

        dominant_idx = np.nanargmax(power[1:]) + 1

        return dominant_idx / len(arr_filled)

    for window in fourier_windows:
        df[f'fft_dom_freq_{window}'] = (
            df['value']
              .shift(1)
              .rolling(window=window, min_periods=window)
              .apply(lambda arr: compute_dominant_freq(arr), raw=True)
        )

    # Групповые статистики по дню недели.
    df['weekday_mean_cum'] = df.groupby('weekday')['value'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    df['weekday_std_cum'] = df.groupby('weekday')['value'].transform(
        lambda x: x.expanding().std().shift(1)
    )
    df['weekday_min_cum'] = df.groupby('weekday')['value'].transform(
        lambda x: x.expanding().min().shift(1)
    )
    df['weekday_max_cum'] = df.groupby('weekday')['value'].transform(
        lambda x: x.expanding().max().shift(1)
    )

    # Дополнительные скользящие статистики (медиана и квантили).
    for window in rolling_windows:
        df[f'rolling_median_{window}'] = df['value'].rolling(window=window).median()
        df[f'rolling_q25_{window}'] = df['value'].rolling(window=window).quantile(0.25)
        df[f'rolling_q75_{window}'] = df['value'].rolling(window=window).quantile(0.75)

    # Целевая переменная.
    df['target'] = df['value']

    # Удаляем столбец raw value и пропуски.
    df.drop(columns=['value', 'weekday'], inplace=True)
    df.dropna(inplace=True)

    return df


def split_time_series(
        df: pd.DataFrame,
        ticker_name: str,
        train_size=0.7,
        val_size=0.2
):
    """
    Разбивает отсортированный временной ряд на обучающую, валидационную и тестовую выборки.

    Parameters:
      df : pd.DataFrame
        Временной ряд.
      ticker_name : str
        Тикер для фильтрации тестовых данных.
      train_size : float
        Пропорция обучающей выборки.
      val_size : float
        Пропорция валидационной выборки.

    Returns:
      df_train, df_val, df_test : pd.DataFrame
        Разделённые данные.
    """
    df_train_list, df_val_list, df_test_list = [], [], []

    for _, group in df.groupby('ticker'):
        group = group.sort_values(by='Date')

        n = len(group)
        train_end = int(n * train_size)
        val_end = train_end + int(n * val_size)

        df_train_list.append(group.iloc[:train_end])
        df_val_list.append(group.iloc[train_end:val_end])
        df_test_list.append(group.iloc[val_end:])

    df_train = pd.concat(df_train_list).reset_index(drop=True)
    df_val = pd.concat(df_val_list).reset_index(drop=True)
    df_test = pd.concat(df_test_list).reset_index(drop=True)

    df_test = df_test[df_test['ticker'] == ticker_name]

    return df_train, df_val, df_test


def read_data(
    file_path: str,
    file_type: str,
    ticker_type: str
) -> pd.DataFrame:
    """
    Чтение и преобразование данных.

    Parameters
    ----------
    file_path : str
        Путь до файла.
    file_type : str
        Тип файла (currency, stock).
    ticker_type : str
        Название типа тикера для считывания.

    Returns
    -------
    df : pd.DataFrame
        Кадр данных.
    """
    df = pd\
        .read_parquet(file_path)\
        .reset_index()\
        .ffill()

    df = pd.melt(frame=df, id_vars='Date', var_name='series_id', value_name='value')

    if file_type == 'currency':
        df['ticker'] = df.apply(
            func=lambda row: row['series_id'].split('=X_')[0],
            axis=1
        )
        df['ticker_type'] = df.apply(
            func=lambda row: row['series_id'].split('=X_')[1],
            axis=1
        )

        df = df[df['ticker_type'] == ticker_type]
        df.drop(columns=['series_id', 'ticker_type'], inplace=True)

    elif file_type == 'stock':
        df['ticker'] = df['series_id']
        df.drop(columns=['series_id'], inplace=True)

    return df
