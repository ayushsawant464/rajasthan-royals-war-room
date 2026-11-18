import os
import time
import numpy as np
import polars as pl
from scipy import stats

def export_universal_calibrated_leaderboards():
    print("=== Exporting Universal Physics & Multi-Cohort Calibrated Leaderboards ===")
    t0 = time.time()
    
    data_dir = '6af458274c1e8_data_pack/Data Pack'
    matches = pl.read_csv(f'{data_dir}/matches.csv', columns=['match_id', 'revised_target', 'win_team_id'])
    bbb_cols = [
        'match_id', 'innings', 'innings_ball_num', 'overid', 'runs', 
        'extras_type', 'extras1_runs', 'extras_2_runs', 'is_wicket', 
        'striker_name', 'bowler_name', 'batting_team_id', 'batting_order', 
        'delivery_style', 'pace_or_spin'
    ]
    bbb = pl.read_csv(f'{data_dir}/ball_by_ball.csv', columns=bbb_cols)

    # 1. Clean Signed Integer Tracking
    bbb = bbb.with_columns([
        (pl.col('runs') + pl.col('extras1_runs') + pl.col('extras_2_runs')).alias('ball_total_runs'),
        (~pl.col('extras_type').is_in(['wide', 'Overstep'])).alias('is_legal_ball'),
        (~pl.col('extras_type').is_in(['wide'])).alias('is_batter_ball'),
        ((pl.col('runs') + pl.col('extras1_runs') + pl.col('extras_2_runs') == 0) & 
         (~pl.col('extras_type').is_in(['wide', 'Overstep']))).alias('is_dot')
    ]).sort(['match_id', 'innings', 'innings_ball_num'])

    bbb = bbb.with_columns([
        pl.col('ball_total_runs').cum_sum().over(['match_id', 'innings']).cast(pl.Int64).alias('cum_runs'),
        pl.col('is_legal_ball').cum_sum().over(['match_id', 'innings']).cast(pl.Int64).alias('cum_legal_balls'),
        pl.col('is_wicket').cum_sum().over(['match_id', 'innings']).cast(pl.Int64).alias('cum_wickets')
    ])
    bbb = bbb.with_columns([
        (pl.col('cum_runs') - pl.col('ball_total_runs')).alias('prev_runs'),
        (pl.col('cum_legal_balls') - pl.when(pl.col('is_legal_ball')).then(1).otherwise(0)).alias('prev_legal_balls'),
        (pl.col('cum_wickets') - pl.when(pl.col('is_wicket')).then(1).otherwise(0)).alias('prev_wickets'),
    ])

    inn1_totals = bbb.filter(pl.col('innings') == 1).group_by('match_id').agg(pl.col('ball_total_runs').sum().alias('inn1_total_runs'))
    matches = matches.join(inn1_totals, on='match_id', how='left').with_columns(
        pl.when(pl.col('revised_target').is_not_null()).then(pl.col('revised_target')).otherwise(pl.col('inn1_total_runs') + 1).alias('target')
    )
    bbb = bbb.join(matches.select(['match_id', 'target', 'win_team_id']), on='match_id', how='left')

    bbb = bbb.with_columns([
        (pl.col('target') - pl.col('prev_runs')).alias('runs_needed'),
        (120 - pl.col('prev_legal_balls')).clip(0, 120).alias('balls_remaining'),
        (10 - pl.col('prev_wickets')).clip(0, 10).alias('wickets_in_hand'),
        (pl.col('win_team_id') == pl.col('batting_team_id')).cast(pl.Float64).alias('chase_won')
    ])

    # Dynamic target in 1st innings based on empirical par
    bbb = bbb.with_columns([
        pl.when(pl.col('innings') == 2).then(pl.col('runs_needed')).otherwise(166 - pl.col('prev_runs')).alias('eff_runs_needed')
    ])
    bbb = bbb.with_columns([
        (pl.col('eff_runs_needed') * 6.0 / pl.col('balls_remaining').clip(1, 120)).alias('rrr')
    ])

    # 2. Continuous Leverage Formulation
    theta = np.array([3.1785, -0.5428, 0.6491, -5.1342])
    X_all = np.column_stack([
        np.ones(len(bbb)),
        bbb['rrr'].clip(-5, 30).to_numpy(),
        bbb['wickets_in_hand'].to_numpy(),
        bbb['balls_remaining'].clip(1, 120).to_numpy() / 120.0
    ])
    z_all = X_all @ theta
    v_all = 1.0 / (1.0 + np.exp(-z_all))
    tau_norm = np.maximum(0.05, bbb['balls_remaining'].clip(1, 120).to_numpy() / 120.0)
    leverage_continuous = (4.0 * v_all * (1.0 - v_all)) / np.sqrt(tau_norm)
    bbb = bbb.with_columns(pl.Series('leverage', leverage_continuous))

    # 3. In-Match Latency & Cohort Labeling
    bbb = bbb.with_columns(
        pl.col('innings_ball_num').cum_count().over(['match_id', 'innings', 'striker_name']).alias('batter_balls_faced'),
        pl.when(pl.col('batting_order') <= 2).then(pl.lit('Top Order (1-2)'))
          .when(pl.col('batting_order') <= 5).then(pl.lit('Middle Order (3-5)'))
          .otherwise(pl.lit('Lower Order (6-11)')).alias('batting_cohort')
    )
    bbb = bbb.with_columns(
        pl.when(pl.col('batter_balls_faced') <= 5).then(pl.lit('Cold (1-5)'))
          .when(pl.col('batter_balls_faced') <= 12).then(pl.lit('Warming (6-12)'))
          .otherwise(pl.lit('Set (13+)')).alias('cold_entry_status')
    )

    # 4. Universal Multi-Level Delivery Physics Baselines
    # A. Batter Baseline: E[runs | overid, delivery_style, cold_entry_status]
    bat_exp_fine = bbb.filter(pl.col('is_batter_ball')).group_by(['overid', 'delivery_style', 'cold_entry_status']).agg([
        pl.col('runs').mean().alias('exp_bat_runs_fine'),
        pl.len().alias('cell_n')
    ])
    bbb = bbb.join(bat_exp_fine, on=['overid', 'delivery_style', 'cold_entry_status'], how='left')

    bat_exp_coarse = bbb.filter(pl.col('is_batter_ball')).group_by(['overid', 'cold_entry_status']).agg(
        pl.col('runs').mean().alias('exp_bat_runs_coarse')
    )
    bbb = bbb.join(bat_exp_coarse, on=['overid', 'cold_entry_status'], how='left')

    bbb = bbb.with_columns(
        pl.when(pl.col('cell_n') >= 15).then(pl.col('exp_bat_runs_fine'))
          .otherwise(pl.col('exp_bat_runs_coarse')).alias('fair_bat_exp_runs')
    )
    bbb = bbb.with_columns((pl.col('runs') - pl.col('fair_bat_exp_runs')).alias('ball_physics_rae'))

    # B. Bowler Baseline: E[ball_total_runs | overid, batting_cohort, cold_entry_status]
    bowl_exp = bbb.group_by(['overid', 'batting_cohort', 'cold_entry_status']).agg(
        pl.col('ball_total_runs').mean().alias('fair_bowl_exp_runs')
    )
    bbb = bbb.join(bowl_exp, on=['overid', 'batting_cohort', 'cold_entry_status'], how='left')
    bbb = bbb.with_columns((pl.col('fair_bowl_exp_runs') - pl.col('ball_total_runs')).alias('ball_physics_rsae'))

    # Continuous Wicket Valuation Manifold lambda(b, w)
    B_vec = bbb['balls_remaining'].clip(1, 120).to_numpy()
    W_vec = bbb['wickets_in_hand'].clip(1, 10).to_numpy()
    lambda_val = np.minimum(28.0, (0.132 * B_vec) * np.power(10.0 / W_vec, 0.70) + (10.0 - W_vec) * 0.85)
    bbb = bbb.with_columns(pl.Series('lambda_val', lambda_val))
    bbb = bbb.with_columns((pl.col('ball_physics_rsae') + pl.col('is_wicket') * pl.col('lambda_val')).alias('ball_physics_win_equity'))

    os.makedirs('outputs', exist_ok=True)

    # 5. Export Calibrated Universal Batters with Regularized Empirical Bayes & Kish N_eff
    print("Computing universal calibrated batters with REML tau^2 and Kish CLT...")
    sigma0_2_bat = float(bbb.filter(pl.col('is_batter_ball'))['ball_physics_rae'].var())
    tau2_bat = 0.02229
    m_prior = 20.0

    bat_df = bbb.filter(pl.col('is_batter_ball'))
    bat_stats = bat_df.group_by('striker_name').agg([
        pl.len().alias('total_deliveries_faced'),
        pl.col('runs').sum().alias('total_runs_scored'),
        pl.col('overid').mean().alias('mean_over_entered'),
        (pl.col('runs').sum() * 100.0 / pl.len()).alias('raw_strike_rate'),
        (pl.col('leverage') * pl.col('runs')).sum().alias('w_runs_sum'),
        (pl.col('leverage') * pl.col('ball_physics_rae')).sum().alias('w_rae_sum'),
        pl.col('leverage').sum().alias('cumulative_leverage_faced'),
        (pl.col('leverage')**2).sum().alias('w2_sum'),
        pl.col('ball_physics_rae').var().alias('sample_var_rae')
    ])

    bat_stats = bat_stats.with_columns([
        (pl.col('w_runs_sum') * 100.0 / pl.col('cumulative_leverage_faced')).alias('continuous_leverage_sr'),
        (pl.col('w_rae_sum') / pl.col('cumulative_leverage_faced')).alias('continuous_physics_rae_per_ball'),
        (pl.col('cumulative_leverage_faced')**2 / pl.col('w2_sum')).alias('kish_eff_deliveries'),
        pl.when(pl.col('mean_over_entered') <= 7.0).then(pl.lit('Top Order (1-2)'))
          .when(pl.col('mean_over_entered') <= 13.0).then(pl.lit('Middle Order (3-5)'))
          .otherwise(pl.lit('Lower Order (6-11)')).alias('primary_cohort')
    ])

    # Regularized sample variance (Efron & Morris empirical Bayes pooling)
    bat_stats = bat_stats.with_columns([
        pl.when(pl.col('total_deliveries_faced') > 1)
          .then(((pl.col('total_deliveries_faced') - 1.0) * pl.col('sample_var_rae').fill_null(sigma0_2_bat) + m_prior * sigma0_2_bat) / (pl.col('total_deliveries_faced') - 1.0 + m_prior))
          .otherwise(sigma0_2_bat).alias('s2_regularized')
    ])

    bat_stats = bat_stats.with_columns([
        (pl.col('s2_regularized') / pl.col('kish_eff_deliveries')).alias('sigma2_i'),
    ])
    bat_stats = bat_stats.with_columns([
        (pl.col('sigma2_i').sqrt()).alias('standard_error'),
        (tau2_bat / (tau2_bat + pl.col('sigma2_i'))).alias('eb_shrinkage_weight')
    ])
    bat_stats = bat_stats.with_columns([
        (pl.col('continuous_physics_rae_per_ball') / pl.col('standard_error')).alias('t_stat'),
        (pl.col('eb_shrinkage_weight') * pl.col('continuous_physics_rae_per_ball')).alias('eb_shrunk_physics_rae'),
        (pl.col('continuous_physics_rae_per_ball') - 1.645 * pl.col('standard_error')).alias('clt_95_lcb_rae')
    ])

    p_vals_bat = [
        float(2 * (1 - stats.norm.cdf(abs(t)))) if t is not None and not np.isnan(t) else 1.0 
        for t in bat_stats['t_stat'].to_list()
    ]
    bat_stats = bat_stats.with_columns([
        pl.Series('p_value', p_vals_bat),
        (pl.col('t_stat') >= 3.50).alias('is_statistically_significant')
    ]).sort('eb_shrunk_physics_rae', descending=True)

    bat_out = 'outputs/continuous_leverage_batters_zero_filter.csv'
    bat_stats.write_csv(bat_out)
    print(f"Saved {len(bat_stats)} universal calibrated batters to {bat_out}")

    # 6. Export Calibrated Universal Bowlers with Regularized Empirical Bayes & Kish N_eff
    print("Computing universal calibrated bowlers with REML tau^2 and Kish CLT...")
    sigma0_2_bowl = float(bbb['ball_physics_win_equity'].var())
    tau2_bowl = 0.02145

    bowl_stats = bbb.group_by('bowler_name').agg([
        pl.len().alias('total_deliveries_bowled'),
        pl.col('ball_total_runs').sum().alias('total_runs_conceded'),
        (pl.col('ball_total_runs').sum() * 6.0 / pl.col('is_legal_ball').sum()).alias('raw_economy'),
        pl.col('is_wicket').sum().alias('total_wickets'),
        pl.col('overid').mean().alias('mean_over_bowled'),
        (pl.col('overid') <= 6).sum().alias('pp_balls'),
        ((pl.col('overid') >= 7) & (pl.col('overid') <= 15)).sum().alias('mid_balls'),
        (pl.col('overid') >= 16).sum().alias('death_balls'),
        (pl.col('leverage') * pl.col('ball_total_runs')).sum().alias('w_runs_sum'),
        (pl.col('leverage') * pl.col('ball_physics_win_equity')).sum().alias('w_equity_sum'),
        (pl.col('leverage') * pl.col('is_legal_ball')).sum().alias('w_legal_sum'),
        (pl.col('leverage') * pl.col('is_wicket')).sum().alias('w_wkt_sum'),
        (pl.col('leverage') * pl.col('is_dot')).sum().alias('w_dot_sum'),
        pl.col('leverage').sum().alias('cumulative_leverage_bowled'),
        (pl.col('leverage')**2).sum().alias('w2_sum'),
        pl.col('ball_physics_win_equity').var().alias('sample_var_equity')
    ])

    bowl_stats = bowl_stats.with_columns([
        (pl.col('w_runs_sum') * 6.0 / pl.col('w_legal_sum')).alias('continuous_leverage_econ'),
        (pl.col('w_equity_sum') / pl.col('cumulative_leverage_bowled')).alias('continuous_physics_win_equity_per_ball'),
        (pl.col('cumulative_leverage_bowled')**2 / pl.col('w2_sum')).alias('kish_eff_deliveries'),
        (pl.col('w_wkt_sum') * 100.0 / pl.col('cumulative_leverage_bowled')).alias('continuous_lev_wicket_pct'),
        (pl.col('w_dot_sum') * 100.0 / pl.col('cumulative_leverage_bowled')).alias('continuous_lev_dot_pct'),
        (pl.col('pp_balls') * 100.0 / pl.col('total_deliveries_bowled')).alias('pp_pct'),
        (pl.col('mid_balls') * 100.0 / pl.col('total_deliveries_bowled')).alias('mid_pct'),
        (pl.col('death_balls') * 100.0 / pl.col('total_deliveries_bowled')).alias('death_pct'),
        pl.when(pl.col('death_balls') * 100.0 / pl.col('total_deliveries_bowled') >= 28.0).then(pl.lit('Death Specialist / Enforcer'))
          .when(pl.col('pp_balls') * 100.0 / pl.col('total_deliveries_bowled') >= 45.0).then(pl.lit('Opening / Powerplay Specialist'))
          .otherwise(pl.lit('Middle Overs Controller')).alias('primary_cohort')
    ])

    bowl_stats = bowl_stats.with_columns([
        pl.when(pl.col('total_deliveries_bowled') > 1)
          .then(((pl.col('total_deliveries_bowled') - 1.0) * pl.col('sample_var_equity').fill_null(sigma0_2_bowl) + m_prior * sigma0_2_bowl) / (pl.col('total_deliveries_bowled') - 1.0 + m_prior))
          .otherwise(sigma0_2_bowl).alias('s2_regularized')
    ])

    bowl_stats = bowl_stats.with_columns([
        (pl.col('s2_regularized') / pl.col('kish_eff_deliveries')).alias('sigma2_i'),
    ])
    bowl_stats = bowl_stats.with_columns([
        (pl.col('sigma2_i').sqrt()).alias('standard_error'),
        (tau2_bowl / (tau2_bowl + pl.col('sigma2_i'))).alias('eb_shrinkage_weight')
    ])
    bowl_stats = bowl_stats.with_columns([
        (pl.col('continuous_physics_win_equity_per_ball') / pl.col('standard_error')).alias('t_stat'),
        (pl.col('eb_shrinkage_weight') * pl.col('continuous_physics_win_equity_per_ball')).alias('eb_shrunk_win_equity'),
        (pl.col('continuous_physics_win_equity_per_ball') - 1.645 * pl.col('standard_error')).alias('clt_95_lcb_equity')
    ])

    p_vals_bowl = [
        float(2 * (1 - stats.norm.cdf(abs(t)))) if t is not None and not np.isnan(t) else 1.0 
        for t in bowl_stats['t_stat'].to_list()
    ]
    bowl_stats = bowl_stats.with_columns([
        pl.Series('p_value', p_vals_bowl),
        (pl.col('t_stat') >= 3.50).alias('is_statistically_significant')
    ]).sort('eb_shrunk_win_equity', descending=True)

    bowl_out = 'outputs/continuous_leverage_bowlers_zero_filter.csv'
    bowl_stats.write_csv(bowl_out)
    print(f"Saved {len(bowl_stats)} universal calibrated bowlers to {bowl_out}")

    t_elapsed = time.time() - t0
    print(f"Universal calibrated leaderboards exported successfully in {t_elapsed:.2f}s.")

if __name__ == '__main__':
    export_universal_calibrated_leaderboards()
