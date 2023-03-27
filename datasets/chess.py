from typing import Tuple
import pandas as pd
from jax import numpy as jnp


def load_chess(start_date: str = '2015-12-31', end_date: str = '2020-01-01', origin_date: str = '2015-12-31')\
    -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict, dict]:
    origin_date = pd.to_datetime(origin_date)
    
    data_url = 'https://raw.githubusercontent.com/huffyhenry/forecasting-candidates/master/data/games.csv'
    data_all = pd.read_csv(data_url)
    data_all['Timestamp'] = pd.to_datetime(data_all['date'], dayfirst=True)
    data_all['Timestamp'] = pd.to_datetime(data_all['Timestamp'], unit='D')
    data_all['TimestampDays'] = (data_all['Timestamp'] - origin_date).astype('timedelta64[D]').astype(int)

    players_arr = pd.unique(pd.concat([data_all['white'], data_all['black']]))
    players_arr.sort()
    players_name_to_id_dict = {a: i for i, a in enumerate(players_arr)}
    players_id_to_name_dict = {i: a for i, a in enumerate(players_arr)}

    data_all = data_all[(data_all['Timestamp'] > start_date) & (data_all['Timestamp'] <= end_date)]
    data_all['whiteID'] = data_all['white'].apply(lambda s: players_name_to_id_dict[s])
    data_all['blackID'] = data_all['black'].apply(lambda s: players_name_to_id_dict[s])

    data_all = data_all.sort_values('Timestamp')
    
    match_times = jnp.array(data_all['TimestampDays'])
    match_player_indices = jnp.array(data_all[['whiteID', 'blackID']])
    
    result_strs = data_all['result'].values
    match_results = jnp.where(result_strs == '1-0', 1, jnp.where(result_strs == '0-1', 2, 0))
    return match_times, match_player_indices, match_results, players_id_to_name_dict, players_name_to_id_dict

