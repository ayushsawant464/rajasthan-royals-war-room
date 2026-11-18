# Visual Guide: Diagnostic Surfaces, Topology, and Verification Plots

This directory contains the finalized, verified quantitative visualizations produced by our evaluation pipeline. Every plot corresponds to an empirical proof or decision surface in our submission report.

---

### 1. `interactive_3d_state_space.html`
* **Type**: Interactive WebGL 3D Visualization (Open in any modern browser).
* **Axes**: 
  * $X$-Axis: Balls Remaining ($t \in [0, 120]$).
  * $Y$-Axis: Wickets Lost ($w \in [0, 10]$).
  * $Z$-Axis: Match Leverage Multiplier $L(t, \Delta R)$ & Dynamic Wicket Valuation $\lambda(t, w)$.
* **Key Insight**: Demonstrates the smooth, singularity-free manifold of match pressure and the parabolic death-phase spike when a team is 9 wickets down.

---

### 2. `plot_17_three_zone_architecture_and_tactical_audit.png`
* **Type**: 4-Panel Tactical & Positional Calibration Suite.
* **Panels**:
  1. *Panel A*: Three-Zone Tactical Architecture (Powerplay vs Middle vs Death strike rates and baseline run expectations).
  2. *Panel B*: Pacer Death Burden (Pacers bowl **79.9%** of death overs; raw economy rates penalize pacers unfairly).
  3. *Panel C*: West Indies Hitters Audit (Andre Russell 171.1 SR / +2.83$\sigma$ vs Romario Shepherd 121.3 SR / $-0.09\sigma$).
  4. *Panel D*: Within-Cohort Positional VORP (Value Over Replacement Player) balancing top, middle, and lower orders.

---

### 3. `plot_16_reverse_swing_physics_and_temporal_audit.png`
* **Type**: Reverse Swing Aerodynamics & Temporal Distribution Audit.
* **Panels**:
  1. *Panel A*: Temporal Concentration (73.4% of all reverse swing occurs in overs 15–20 where average par is 10.5+ RPO).
  2. *Panel B*: Pitch Length Distribution (Comparing yorker, full, good, and full toss concessions).
  3. *Panel C*: Run Concession Decomposition (Proving high runs conceded are due to the death-overs match situation, NOT delivery failure).
  4. *Panel D*: Wicket Hazard Ratio (Reverse swing generates 2.4× higher wicket hazard when bowled on a yorker length).

---

### 4. `plot_15_universal_physics_and_cohort_calibration.png`
* **Type**: Delivery Physics Trajectory & Cohort Response Surfaces.
* **Features**: Joint bivariate surfaces showing expected run yield as a function of release speed (km/h) and pitch length across seam, swing, and spin archetypes.

---

### 5. `plot_14_cohort_surfaces_and_statistical_invariants.png`
* **Type**: Central Limit Theorem (CLT) Funnel & Cohort Distribution.
* **Features**: Scatter plot of all 885 batters showing the variance funnel collapsing towards true mean as Kish effective sample size increases, alongside Empirical Bayes shrinkage trajectories.

---

### 6. `plot_13_bowler_exact_boundaries_and_continuous_models.png`
* **Type**: Bowler Continuous Leverage & Boundary Verification.
* **Features**: Bowler equity distributions, phase burden curves, and zero-filter shrinkage stability for 646 bowlers.

---

### 7. `plot_12_exact_boundaries_and_correlation_distribution.png`
* **Type**: Batter Exact Boundary Conditions & Parameter Correlation Matrix.
* **Features**: 10×10 correlation matrix verifying orthogonal independence of model features and boundary limits as balls remaining $\to 0$.

---

### 8. `plot_11_death_wicket_and_opener_debiasing.png`
* **Type**: Death Wicket Manifold $\lambda(t, w)$ & Powerplay Field-Restriction Debiasing.
* **Features**: The mathematical proof showing why early wickets are worth 16 runs while death wickets drop to 4 runs (unless 9 down), and how opener run rates are normalized against fielding restrictions.

---

### 9. `plot_10_decision_verification_surfaces.png`
* **Type**: Master Decision Verification Surfaces.
* **Features**: 3D response curves of Win Probability Added ($\Delta WP$) versus Required Run Rate ($RRR$) and current Net Run Rate.

---

### 10. `plot_9_quant_derivative_pricing_surfaces.png`
* **Type**: Financial Quantitative Derivative Analogy Surfaces.
* **Features**: Volatility smiles, leverage Greeks ($\partial WP / \partial R$), and gamma curvature across match states.

---

### 11. `plot_8_win_probability_topology_and_phase_diagram.png`
* **Type**: Win Probability Topology & Phase Diagram.
* **Features**: Contour map showing the transition boundaries between high-leverage and low-leverage match phases.
