from typing import Tuple
import pandas as pd
from jax import numpy as jnp


def load_epl(start_date: str = '2018-07-30',
             end_date: str = '2022-07-01',
             origin_date: str = '2018-07-30')\
        -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict, dict]:
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    origin_date = pd.to_datetime(origin_date)
    
    cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
    data = []
    for year in range(start_date.year, end_date.year):
        year_str = str(year)[-2:]
        nyear_str = str(year + 1)[-2:]
        data.append(pd.read_csv(f'https://www.football-data.co.uk/mmz4281/{year_str}{nyear_str}/E0.csv')[cols])
    data_all = pd.concat(data)

    data_all = data_all.dropna()
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

    match_results = jnp.where(home_goals > away_goals, 1,
                              jnp.where(home_goals < away_goals, 2, 0))

    return match_times, match_player_indices, match_results, players_id_to_name_dict, players_name_to_id_dict


def load_international(start_date: str = '2018-07-30',
                       end_date: str = '2022-07-01',
                       origin_date: str = '2018-07-30')\
        -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict, dict]:
    origin_date = pd.to_datetime(origin_date)

    data_url = 'https://raw.githubusercontent.com/martj42/international_results/master/results.csv'
    data_all = pd.read_csv(data_url)

    data_all['Timestamp'] = pd.to_datetime(data_all['date'])
    data_all['Timestamp'] = pd.to_datetime(data_all['Timestamp'], unit='D')
    data_all['TimestampDays'] = (
        data_all['Timestamp'] - origin_date).astype('timedelta64[D]').astype(int)

    players_arr = pd.unique(pd.concat([data_all['home_team'], data_all['away_team']]))
    players_arr.sort()
    players_name_to_id_dict = {a: i for i, a in enumerate(players_arr)}
    players_id_to_name_dict = {i: a for i, a in enumerate(players_arr)}

    data_all = data_all[(data_all['Timestamp'] > start_date) & (data_all['Timestamp'] <= end_date)]
    data_all['HomeTeamID'] = data_all['home_team'].apply(lambda s: players_name_to_id_dict[s])
    data_all['AwayTeamID'] = data_all['away_team'].apply(lambda s: players_name_to_id_dict[s])

    match_times = jnp.array(data_all['TimestampDays'])
    match_player_indices = jnp.array(data_all[['HomeTeamID', 'AwayTeamID']])

    home_goals = jnp.array(data_all['home_score'])
    away_goals = jnp.array(data_all['away_score'])

    match_results = jnp.where(home_goals > away_goals, 1,
                              jnp.where(home_goals < away_goals, 2, 0))

    return match_times, match_player_indices, match_results, players_id_to_name_dict, players_name_to_id_dict
