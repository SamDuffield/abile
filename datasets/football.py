from typing import Tuple
import pandas as pd
from jax import numpy as jnp


def load_epl(start_date: str = '2018-07-30', end_date: str = '2022-07-01', origin_date: str = '2018-07-30')\
    -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict, dict]:
    origin_date = pd.to_datetime(origin_date)

    cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
    data_18_19 = pd.read_csv('datasets/epl_2018_2019.csv')[cols]
    data_19_20 = pd.read_csv('datasets/epl_2019_2020.csv')[cols]
    data_20_21 = pd.read_csv('datasets/epl_2020_2021.csv')[cols]
    data_21_22 = pd.read_csv('datasets/epl_2021_2022.csv')[cols]

    data_all = pd.concat([data_18_19, data_19_20, data_20_21, data_21_22])
    data_all['Timestamp'] = pd.to_datetime(data_all['Date'], dayfirst=True)
    data_all['Timestamp'] = pd.to_datetime(data_all['Timestamp'], unit='D')
    data_all['TimestampDays'] = (data_all['Timestamp'] - origin_date).astype('timedelta64[D]').astype(int)

    players_arr = pd.unique(pd.concat([data_all['HomeTeam'], data_all['AwayTeam']]))
    players_arr.sort()
    players_name_to_id_dict = {a: i for i, a in enumerate(players_arr)}
    players_id_to_name_dict = {i: a for i, a in enumerate(players_arr)}

    data_all = data_all[(data_all['Timestamp'] > start_date) & (data_all['Timestamp'] <= end_date)]
    data_all['HomeTeamID'] = data_all['HomeTeam'].apply(lambda s: players_name_to_id_dict[s])
    data_all['AwayTeamID'] = data_all['AwayTeam'].apply(lambda s: players_name_to_id_dict[s])

    match_times = jnp.array(data_all['TimestampDays'])
    match_player_indices = jnp.array(data_all[['HomeTeamID', 'AwayTeamID']])

    home_goals = jnp.array(data_all['FTHG'])
    away_goals = jnp.array(data_all['FTAG'])

    match_results = jnp.where(home_goals > away_goals, 1,\
        jnp.where(home_goals < away_goals, 2, 0))

    return match_times, match_player_indices, match_results, players_id_to_name_dict, players_name_to_id_dict