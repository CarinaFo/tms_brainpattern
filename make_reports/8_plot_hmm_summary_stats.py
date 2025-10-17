import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import os
from pathlib import Path

# run in base python (3.12) on lucky3 (linux)

# set working directory
os.chdir(Path("/home/carinaf"))

base_dir = os.getcwd()

n_states = 8

hmm_dir = Path(f'{base_dir}/tms_mdd/prepared_data_80patients_giles_newmodel')
csv_path = Path(f"{hmm_dir}/hmm_demo_quest_{n_states}.csv")

save_path = f'{hmm_dir}/tms_mdd/{n_states}_allsessions_plots'

if not os.path.exists(save_path):
    os.makedirs(save_path)

#results = run_complete_analysis(csv_path, n_states=n_states, save_plots=False)

#df = load_and_prep_data(csv_path)
#correlations = plot_hads_fo_correlations(df, save_plots=True)
#predictions = predict_symptom_improvement(df, target_states=[3, 4], save_plots=True)

def load_and_prep_data(csv_path, n_states=6, plot_stuff=1):
    """Load and preprocess the HMM data"""
    df = pd.read_csv(csv_path)
    
    # Exclude problematic patients
    exclude_ids = ['144R', '127', '182']
    df = df[~df['patient'].isin(exclude_ids)]
    df = df[~df["patient"].str.contains("R")]
    
    print(f'Analyzing {df["patient"].nunique()} patients')
    
    # Recode states (state 0 is weird)
    df['state'] = df['state'] + 1
    
    # Fill missing values
    for col in ['age', 'gender', 'responder', 'group']:
        df[col] = df.groupby('patient')[col].transform('first')
    
    # Convert to categorical
    for col in ['patient', 'session', 'tms', 'state', 'group', 'responder']:
        df[col] = df[col].astype('category')
    
    # Apply logit transform to fo
    eps = 1e-6
    df['fo_logit'] = np.log((df['fo'] + eps) / (1 - df['fo'] - eps))

    if plot_stuff:

        plot_clinical_scores_over_sessions_violin(df)

        plot_hmm_measures_violin(df, state=4)

    return df

def calculate_symptom_improvement(df):
    """Calculate symptom improvement from session 1 to 3"""
    # Collapse data: average scores per patient per session
    df_clin = (
        df.groupby(['patient', 'session'], as_index=False)
        [['dep_hads', 'anxiety_hads', 'madrs_score', 'hama_score']]
        .mean()
    )
    
    scores = ['madrs_score', 'dep_hads', 'hama_score']
    
    # Get session 1 and 3 data
    df_ses1 = df_clin[df_clin['session'] == 1].set_index('patient')
    df_ses3 = df_clin[df_clin['session'] == 3].set_index('patient')
    
    # Calculate relative improvement
    rel_change = pd.DataFrame(index=df_ses1.index)
    for score in scores:
        diff = df_ses1[score] - df_ses3[score]
        rel_change[score] = diff / df_ses1[score]
    
    return rel_change.reset_index()


def plot_hads_fo_correlations(df, n_states=n_states, save_plots=False):
    """Plot correlations between HADS depression and FO for each state at session 1"""
    # Filter for session 1, pre-TMS
    df_session1pre = df[(df.session == 1) & (df.tms == 'pre')].copy()
    
    states = sorted(df['state'].unique())
    
    # Set up subplot grid
    if n_states == 12:
        fig, axes = plt.subplots(3, 4, figsize=(15, 12))
    elif n_states == 10:
        fig, axes = plt.subplots(2, 5, figsize=(15, 12))
    else:
        fig, axes = plt.subplots(2, 4, figsize=(15, 12))

    axes = axes.flatten()
    
    correlations = []
    
    for i, state in enumerate(states):
        df_state = df_session1pre[df_session1pre['state'] == state]
        
        if len(df_state) == 0:
            axes[i].text(0.5, 0.5, 'No data', ha='center', va='center', 
                        transform=axes[i].transAxes)
            axes[i].set_title(f'State {state}')
            continue
        
        # Calculate Spearman correlation
        rho, p = spearmanr(df_state['dep_hads'], df_state['fo_logit'], 
                          nan_policy='omit')
        correlations.append({'state': state, 'rho': rho, 'p': p})
        
        # Create scatter plot
        ax = axes[i]
        ax.scatter(df_state['dep_hads'], df_state['fo_logit'], alpha=0.7, s=30)
        
        # Add regression line for visualization
        valid_data = df_state[['dep_hads', 'fo_logit']].dropna()
        if len(valid_data) > 1:
            m, b = np.polyfit(valid_data['dep_hads'], valid_data['fo_logit'], 1)
            ax.plot(valid_data['dep_hads'], m*valid_data['dep_hads'] + b, 
                   color='red', linewidth=2)
        
        # Customize plot
        ax.set_title(f'State {state}\nρ = {rho:.3f}, p = {p:.3f}', 
                    fontweight='bold')
        ax.set_xlabel('HADS Depression Score')
        ax.set_ylabel('FO (logit) Session 1 Pre-TMS')
        ax.grid(True, alpha=0.3)
        
        # Color title based on significance
        if p < 0.05:
            ax.title.set_color('red')
    
    # Remove empty subplots
    for j in range(len(states), len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle('HADS Depression vs Fractional Occupancy Correlations\n(Session 1, Pre-TMS)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('hads_fo_correlations_by_state.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print correlation summary with FDR correction
    corr_df = pd.DataFrame(correlations)
    if len(corr_df) > 0:
        rejected, pvals_corrected, _, _ = multipletests(corr_df['p'], 
                                                       alpha=0.05, method='fdr_bh')
        corr_df['p_fdr'] = pvals_corrected
        corr_df['significant_fdr'] = rejected
        
        print("\n=== HADS-FO CORRELATION SUMMARY ===")
        print(corr_df.round(3))
    
    # now run linex mixed effects models
    dvs = ['fo_logit']  # dependent variables
    ivs = ['dep_hads']

    all_results = {}

    for dv in dvs:
        pvals = []

        # Fit models for each state
        for state in df['state'].unique():
            df_state = df_session1pre[df_session1pre['state'] == state]

            df_state = df_state.reset_index(drop=True)


            if ivs[0] == 'hama_score' or ivs[0] == 'madrs_score':
                df_state = df[df['state'] == state].dropna(subset=['hama_score'])

                # Drop unused session levels if categorical
                if pd.api.types.is_categorical_dtype(df_state['session']):
                    df_state['session'] = df_state['session'].cat.remove_unused_categories()

                # Skip if not enough data
                if df_state.shape[0] < 3 or df_state['session'].nunique() < 1:
                    continue

            # random intercepts
            model = smf.ols(f"{dv} ~ {ivs[0]} + gender + age + I(age**2) ", 
                                df_state)
            result = model.fit()
            print(result.summary())

            # Get p-value for anxiety_hads
            pvals.append(result.pvalues[ivs[0]])

        # Apply FDR correction for this DV
        rejected, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

        # Store results
        all_results[dv] = list(zip(df['state'].unique(), pvals, pvals_corrected, rejected))

    # Print nicely
    for dv, res_list in all_results.items():
        print(f"\n=== Results for {dv} ===")
        for state, pval, pval_corr, sig in res_list:
            print(f"State: {state}, raw p: {pval:.4f}, FDR p: {pval_corr:.4f}, significant: {sig}")

    return corr_df


def predict_symptom_improvement(df, target_states=None, save_plots=False):
    """Predict symptom improvement using FO at session 1"""
    
    if target_states is None:
        target_states = [1, 2, 8]  # Default states of interest
    
    # Get session 1 pre-TMS data
    df_session1 = df[(df.session == 1) & (df.tms == 'pre')].copy()
    
    # Calculate symptom improvement
    symptom_change = calculate_symptom_improvement(df)
    
    # Pivot FO data to wide format
    df_wide = df_session1.pivot_table(
        index="patient",
        columns="state", 
        values="fo_logit"
    ).reset_index()
    
    # Merge with baseline scores and outcomes
    baseline_scores = df_session1.groupby('patient')[
        ['dep_hads', 'madrs_score', 'age', 'gender', 'responder', 'group']
    ].first().reset_index()
    
    df_analysis = df_wide.merge(baseline_scores, on='patient')
    df_analysis = df_analysis.merge(symptom_change[['patient', 'dep_hads', 'madrs_score']], 
                                   on='patient', suffixes=('_baseline', '_change'))
    
    # Rename FO columns
    fo_cols = [col for col in df_analysis.columns if isinstance(col, int)]
    df_analysis = df_analysis.rename(columns={s: f'fo_state{s}' for s in fo_cols})
    
    results = {}
    
    for state in target_states:
        fo_col = f'fo_state{state}'
        if fo_col not in df_analysis.columns:
            continue
            
        # Clean data
        analysis_data = df_analysis[['patient', fo_col, 'dep_hads_change', 'madrs_score_change', 
                                    'madrs_score_baseline', 'dep_hads_baseline', 'age', 'gender',
                                     'responder', 'group']].dropna()
        
        if len(analysis_data) < 10:
            print(f"Insufficient data for state {state}")
            continue
        
        # Fit regression model
        formula = f"dep_hads_change ~ {fo_col} * responder"
        model = smf.ols(formula, data=analysis_data).fit()
        
        results[state] = {
            'model': model,
            'data': analysis_data,
            'fo_coef': model.params[fo_col],
            'fo_pval': model.pvalues[fo_col],
            'r_squared': model.rsquared
        }
        
        print(f"\n=== PREDICTION MODEL: STATE {state} ===")
        print(f"FO coefficient: {model.params[fo_col]:.4f}")
        print(f"FO p-value: {model.pvalues[fo_col]:.4f}")
        print(f"R-squared: {model.rsquared:.3f}")
    
    # Create prediction plots
    if results and save_plots:
        n_states_plot = len(results)
        fig, axes = plt.subplots(1, n_states_plot, figsize=(6*n_states_plot, 5))
        if n_states_plot == 1:
            axes = [axes]
        
        for i, (state, result) in enumerate(results.items()):
            ax = axes[i]
            data = result['data']
            fo_col = f'fo_state{state}'
            
            # Scatter plot
            ax.scatter(data[fo_col], data['dep_hads_change'], alpha=0.7)
            
            # Regression line
            x_range = np.linspace(data[fo_col].min(), data[fo_col].max(), 100)
            pred_data = data.iloc[:1].copy()  # Template row
            predictions = []
            
            for x_val in x_range:
                pred_data[fo_col] = x_val
                pred = result['model'].predict(pred_data)[0]
                predictions.append(pred)
            
            ax.plot(x_range, predictions, 'r-', linewidth=2)
            
            # Customize plot
            ax.set_xlabel(f'FO State {state} (logit) - Session 1')
            ax.set_ylabel('HADS Depression Change\n(Session 1 → 3)')
            ax.set_title(f'State {state}\nβ = {result["fo_coef"]:.3f}, p = {result["fo_pval"]:.3f}')
            ax.grid(True, alpha=0.3)
            
            # Color title based on significance
            if result['fo_pval'] < 0.05:
                ax.title.set_color('red')
        
        plt.suptitle('Fractional Occupancy Predicting Symptom Improvement', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('fo_symptom_prediction.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    return results


def run_complete_analysis(csv_path, n_states=6, target_states=None, save_plots=False):
    """Run the complete analysis pipeline"""
    
    print("=== LOADING AND PREPROCESSING DATA ===")
    df = load_and_prep_data(csv_path, n_states)
    
    print("\n=== ANALYZING HADS-FO CORRELATIONS ===")
    correlations = plot_hads_fo_correlations(df, n_states, save_plots)
    
    print("\n=== PREDICTING SYMPTOM IMPROVEMENT ===")
    if target_states is None:
        target_states = [1, 2, 3, 4, 5, 6] if n_states == 6 else [1, 2, 8]
    
    prediction_results = predict_symptom_improvement(df, target_states, save_plots)
    
    return {
        'data': df,
        'correlations': correlations,
        'predictions': prediction_results
    }



def plot_clinical_scores_over_sessions_violin(df_clin, figsize=(12, 10), save_plots=False):

    """
    Plot clinical scores over sessions using violin plots in 2x2 grid
    Handles HAMA and MADRS having only sessions 1 and 3
    
    Parameters:
    df_clin: DataFrame with clinical scores and session data
    figsize: tuple, figure size
    save_plots: bool, whether to save the plot
    """
    # subset dataframe
    df_onestate=df_clin[(df_clin.state == 1) & (df_clin.tms == 'pre')]
    df_clin = df_onestate
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    score_cols = ['dep_hads', 'anxiety_hads', 'madrs_score', 'hama_score']
    score_labels = ['Depression (HADS)', 'Anxiety (HADS)', 'MADRS Score', 'HAMA Score']
    axes = axes.flatten()
    
    # Set color palette
    colors = sns.color_palette("viridis", len(df_clin['session'].unique()))
    
    for ax, score, label in zip(axes, score_cols, score_labels):
        # Check if column exists and has data
        if score not in df_clin.columns:
            ax.text(0.5, 0.5, f'{score} not found', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(label, fontweight='bold')
            continue
            
        # Remove rows with missing data for this score
        score_data = df_clin[df_clin[score].notna()].copy()
        
        if len(score_data) == 0:
            ax.text(0.5, 0.5, f'No data for {score}', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(label, fontweight='bold')
            continue
        
        # Get available sessions for this score
        available_sessions = sorted(score_data['session'].unique())
        
        # Create violin plot
        sns.violinplot(data=score_data, x='session', y=score, ax=ax, 
                      palette=colors, inner='quart')
        
        # Customize plot
        ax.set_title(label, fontweight='bold')
        ax.set_xlabel('Session')
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add sample size annotations
        session_counts = score_data.groupby('session').size()
        y_min, y_max = ax.get_ylim()
        y_text = y_min + 0.02 * (y_max - y_min)
        
        for i, session in enumerate(available_sessions):
            count = session_counts[session]
            ax.text(i, y_text, f'n={count}', 
                   ha='center', va='bottom', fontsize=8, alpha=0.7)
        
        # Add mean trend line
        session_means = score_data.groupby('session')[score].mean()
        # Use the actual position indices from the violin plot
        session_positions = range(len(session_means))
        ax.plot(session_positions, session_means.values, 
               color='red', linewidth=2, marker='o', markersize=4, 
               alpha=0.8, label='Mean')
        
        # Add special note for HAMA and MADRS (sessions 1 and 3 only)
        if score in ['hama_score', 'madrs_score'] and len(available_sessions) == 2:
            ax.text(0.02, 0.98, 'Sessions 1 & 3 only', 
                   transform=ax.transAxes, fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   verticalalignment='top')
        
        ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('clinical_scores_violins_over_sessions.png', 
                   dpi=300, bbox_inches='tight')
    
    plt.show()
    


def plot_hmm_measures_violin(df, state: int = 1, figsize=(12, 6), save_plots=False):
    """
    Plot HMM measures (fo_logit, sr) over sessions using violin plots
    Shows both pre and post TMS conditions
    
    Parameters:
    df: DataFrame with HMM data
    state: int, which state to analyze (default 3)
    figsize: tuple, figure size
    save_plots: bool, whether to save the plot
    """
    
    # Filter for the specified state
    state_data = df[df['state'] == state].copy()
    
    if len(state_data) == 0:
        print(f"No data found for state {state}")
        return
    
    # Average across trials for each patient/session/tms combination
    df_hmm = (
        state_data
        .groupby(['patient', 'session', 'tms'], as_index=False)
        [['fo_logit', 'sr']]
        .mean()
    )
    
    print(f"HMM data for state {state}:")
    print(f"Patients: {df_hmm['patient'].nunique()}")
    print(f"Sessions: {sorted(df_hmm['session'].unique())}")
    print(f"TMS conditions: {sorted(df_hmm['tms'].unique())}")
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    score_cols = ['fo_logit', 'sr']
    score_labels = ['Fractional Occupancy (logit)', 'Switch Rate']
    
    # Set color palette for TMS conditions
    tms_conditions = sorted(df_hmm['tms'].unique())
    colors = sns.color_palette("Set1", len(tms_conditions))
    
    for ax, score, label in zip(axes, score_cols, score_labels):
        # Check if column exists and has data
        if score not in df_hmm.columns:
            ax.text(0.5, 0.5, f'{score} not found', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(label, fontweight='bold')
            continue
            
        # Remove rows with missing data for this score
        score_data = df_hmm[df_hmm[score].notna()].copy()
        
        if len(score_data) == 0:
            ax.text(0.5, 0.5, f'No data for {score}', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(label, fontweight='bold')
            continue
        
        # Create violin plot
        sns.violinplot(data=score_data, x='session', y=score, hue='tms', 
                      ax=ax, palette=colors, inner='quart', split=False)
        
        # Customize plot
        ax.set_title(f'{label} - State {state}', fontweight='bold')
        ax.set_xlabel('Session')
        ax.set_ylabel(label.split('(')[0].strip())  # Remove units from ylabel
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add sample size annotations
        session_tms_counts = score_data.groupby(['session', 'tms']).size().reset_index(name='count')
        y_min, y_max = ax.get_ylim()
        y_text = y_min + 0.02 * (y_max - y_min)
        
        sessions = sorted(score_data['session'].unique())
        for i, session in enumerate(sessions):
            session_subset = session_tms_counts[session_tms_counts['session'] == session]
            if len(session_subset) > 0:
                count_text = '/'.join([f"n={row['count']}" for _, row in session_subset.iterrows()])
                ax.text(i, y_text, count_text, 
                       ha='center', va='bottom', fontsize=8, alpha=0.7)
        
        # Add mean trend lines for each TMS condition
        for tms_condition in tms_conditions:
            tms_data = score_data[score_data['tms'] == tms_condition]
            if len(tms_data) > 0:
                session_means = tms_data.groupby('session')[score].mean()
                # Map session numbers to violin plot positions
                session_positions = [sessions.index(s) for s in session_means.index]
                
                ax.plot(session_positions, session_means.values, 
                       color=colors[tms_conditions.index(tms_condition)], 
                       linewidth=2, marker='o', markersize=4, 
                       alpha=0.8, linestyle='--')
        
        # Position legend
        ax.legend(title='TMS', loc='upper right', fontsize=8)
    
    # Overall title
    fig.suptitle(f'HMM Measures Over Sessions - State {state}', 
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f'hmm_measures_violins_state_{state}.png', 
                   dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print summary statistics
    print(f"\n=== HMM MEASURES SUMMARY - STATE {state} ===")
    for score, label in zip(score_cols, score_labels):
        if score in df_hmm.columns:
            score_data = df_hmm[df_hmm[score].notna()]
            if len(score_data) > 0:
                print(f"\n--- {label} ({score}) ---")
                
                summary = score_data.groupby(['session', 'tms'])[score].agg([
                    'count', 'mean', 'std', 'median', 
                    lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)
                ]).round(3)
                summary.columns = ['n', 'mean', 'std', 'median', 'Q1', 'Q3']
                print(summary)
