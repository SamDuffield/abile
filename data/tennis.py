from typing import Tuple
import pandas as pd
from jax import numpy as jnp


def consolidate_name_strings(name_series):
    return name_series.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')


def clean_tennis_data(tennis_df_in, origin_date_str, name_to_id_dict):
    origin_date = pd.to_datetime(origin_date_str)
    tennis_df = tennis_df_in.copy()
    tennis_df['Timestamp'] = pd.to_datetime(tennis_df['Date'], dayfirst=True)
    tennis_df['Timestamp'] = pd.to_datetime(tennis_df['Timestamp'], unit='D')
    tennis_df['TimestampDays'] = (tennis_df['Timestamp'] - origin_date).astype('timedelta64[D]').astype(int)
    tennis_df = tennis_df.sort_values('Timestamp')
    tennis_df.reset_index()
    tennis_df['Winner'] = consolidate_name_strings(tennis_df['Winner'])
    tennis_df['Loser'] = consolidate_name_strings(tennis_df['Loser'])
    tennis_df['WinnerID'] = tennis_df['Winner'].apply(lambda s: name_to_id_dict[s])
    tennis_df['LoserID'] = tennis_df['Loser'].apply(lambda s: name_to_id_dict[s])
    return tennis_df



def load_wta(start_date: str = '2020-12-31', end_date: str = '2023-01-01')\
    -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict, dict]:
    data_2021 = pd.read_csv('data/wta_2021.csv')
    data_2022 = pd.read_csv('data/wta_2022.csv')

    data_all = pd.concat([data_2021, data_2022])

    players_arr = pd.unique(pd.concat([consolidate_name_strings(data_all['Winner']),\
        consolidate_name_strings(data_all['Loser'])]))
    players_arr.sort()
    players_name_to_id_dict = {a: i for i, a in enumerate(players_arr)}
    players_id_to_name_dict = {i: a for i, a in enumerate(players_arr)}

    data_all = clean_tennis_data(data_all, '2020-12-31', players_name_to_id_dict)
    data_all = data_all[(data_all['Timestamp'] > start_date) & (data_all['Timestamp'] <= end_date)]

    train_match_times = jnp.array(data_all['TimestampDays'])
    train_match_player_indices = jnp.array(data_all[['WinnerID', 'LoserID']])
    train_match_results = jnp.ones_like(train_match_times)

    return train_match_times, train_match_player_indices, train_match_results, players_id_to_name_dict, players_name_to_id_dict
