import logging
from typing import Dict

import numpy as np
import pandas as pd
import math

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

logging.basicConfig(level=logging.INFO)


class ModelDailyEvaluator:
    """
    This class takes a model and does walk-forward test
    """

    def __init__(self,
                 ponds: Dict[str, pd.DataFrame],  # Dict of dfs with data, a df for each pond
                 encode_ponds: Dict
                 ):
        self.ponds = ponds
        self.encode_ponds = encode_ponds
        self.max_days = 0
        self.cols = list(ponds.values())[0].columns
        self.results = {}

        for pond in self.ponds.keys():
            pond_df = self.ponds[pond]
            print(f'days for pond: {pond} = {pond_df.shape[0]}')
            self.max_days = max(self.max_days, pond_df['days_passed'].iloc[-1])

    def evaluate_all_ponds_daily(self,
                                 model,
                                 pond_names=[],
                                 n_feat_rfe=10,
                                 max_window_size=-1  # Max size of the dataset. -1 = no limit on window size
                                 ):
        """
        Train model on window size on data for all ponds, then predict the next day
        """
        if not pond_names:
            pond_names = self.ponds.keys()

        # Eliminate features
        RFE_columns = []
        y = pd.Series(name='y')
        X = pd.DataFrame(columns=list(self.cols).remove('y'))
        for pond in pond_names:
            df = self.ponds[pond]
            y_to_add = df.loc[(df['days_passed'] >= 0) & (df['days_passed'] < self.max_days), 'y']
            X_to_add = df.loc[(df['days_passed'] >= 0) & (df['days_passed'] < self.max_days)].drop(columns='y')
            if len(y_to_add) and len(X_to_add):
                y = pd.concat([y, y_to_add], ignore_index=True)
                X = pd.concat([X, X_to_add], ignore_index=True)
        if len(y) and len(X):
            rfe = RFE(estimator=model, n_features_to_select=n_feat_rfe)
            rfe.fit(X, y)
            RFE_columns = X.columns[rfe.support_]
            if 'pond_no' not in RFE_columns and len(pond_names) > 1:
                RFE_columns = RFE_columns.append(pd.Index(['pond_no']))

        self.results = {}
        for pond_name in pond_names:
            self.results[pond_name] = {
                'test': [],
                'pred': [],
                'err': [],
                'days_passed': []
            }

        start = 0

        for i in range(self.max_days + 1):
            end = i

            if ((end - start) > max_window_size - 1) and (max_window_size > 0):
                start = end - max_window_size + 1

            # Create X and y for all ponds (all ponds data in one df).
            y = pd.Series(name='y')
            X = pd.DataFrame(columns=list(RFE_columns))
            y_test = pd.Series(name='y')
            X_test = pd.DataFrame(columns=list(RFE_columns))
            for pond in pond_names:
                df = self.ponds[pond]
                y_to_add = df.loc[(df['days_passed'] >= start) & (df['days_passed'] < end), 'y']
                X_to_add = df.loc[(df['days_passed'] >= start) & (df['days_passed'] < end), RFE_columns]
                y_test_to_add = df.loc[df['days_passed'] == end, 'y']
                X_test_to_add = df.loc[df['days_passed'] == end, RFE_columns]
                if len(y_to_add) and len(X_to_add):
                    y = pd.concat([y, y_to_add], ignore_index=True)
                    X = pd.concat([X, X_to_add], ignore_index=True)
                if len(y_test_to_add) and len(X_test_to_add):
                    y_test = pd.concat([y_test, y_test_to_add], ignore_index=True)
                    X_test = pd.concat([X_test, X_test_to_add], ignore_index=True)

            if len(y) and len(X):
                model.fit(X, y)
                y_pred = model.predict(X_test)

                # Fetch results for each pond
                for pond_name in pond_names:
                    pond_code = self.encode_ponds[pond_name]
                    # row number that corresponds to a certain pond
                    idx = X_test[X_test['pond_no'] == pond_code].index.tolist()
                    if idx:
                        idx = idx[0]
                        pond_y_test = y_test.loc[idx]
                        pond_y_pred = y_pred[idx]
                        self.results[pond_name]['test'].append(pond_y_test)
                        self.results[pond_name]['pred'].append(pond_y_pred)
                        self.results[pond_name]['err'].append(abs(pond_y_test - pond_y_pred))
                        self.results[pond_name]['days_passed'].append(
                            X_test.loc[idx, 'days_passed']
                        )

            # Feature iportances
            if i == self.max_days:  # we have trained on the last window
                if isinstance(model, RandomForestRegressor):
                    # Get feature importances
                    importances = model.feature_importances_
                    features = X.columns

                    # Create a DataFrame to hold the importances and features
                    importance_df = pd.DataFrame({'feature': features, 'importance': importances})

                    # Sort the DataFrame by importance in descending order
                    importance_df = importance_df.sort_values('importance', ascending=False)

                    # Plot feature importances
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x='importance', y='feature', data=importance_df)
                    plt.title('Feature Importance')
                    plt.show()
                    features_list = []
                    for ft in importance_df['feature']:
                        features_list.append(ft)
                    print(f' Features selected by RFE: {features_list}')

        return self.results


    def plot_results(self, show_pic=True, save_pic=True):
        N = len(self.results.keys())  # Total number of plots
        cols = 2  # Define number of columns
        rows = math.ceil(N / cols)  # Calculate number of rows needed

        # Create subplots
        fig, axs = plt.subplots(rows, cols, figsize=(12, 6 * rows))

        # Flatten the axes for easier iteration
        axs = axs.ravel()

        for idx, pond in enumerate(self.results.keys()):
            res = self.results[pond]

            RMSE = mean_squared_error(res['test'], res['pred'], squared=False)
            r2_value = r2_score(res['test'], res['pred'])

            print(f'pond: {pond}, RMSE: {RMSE:.2f}, R2: {r2_value:.2f}')

            axs[idx].scatter(res['days_passed'], res['pred'], color='#F1A313', marker='o', s=20)
            axs[idx].plot(
                res['days_passed'], res['pred'],
                label='Predicted Data', color='#F1A313', linestyle='-', linewidth=2
            )

            axs[idx].scatter(res['days_passed'], res['test'], color='#2F79DE', s=20)
            axs[idx].plot(res['days_passed'], res['test'],
                          label='Real Data', color='#2F79DE', linestyle='-', linewidth=2
                          )

            axs[idx].plot(res['days_passed'], res['err'], label='absolute error', color='#13F145')

            axs[idx].set_xlabel('Days Passed', fontsize=10)
            axs[idx].set_ylabel('O2 Level', fontsize=14)
            axs[idx].set_title(f'O2 Morning Level Prediction for Pond {pond}', fontsize=12, y=1.02)
            axs[idx].legend(loc='best', fontsize=12)
            axs[idx].grid(True, linestyle='--', alpha=0.5)

            axs[idx].text(
                0.1, 0.9, f'RMSE: {RMSE}',
                horizontalalignment='left',
                verticalalignment='center',
                transform=axs[idx].transAxes
            )

        # Adjust the layout
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4)
        if save_pic:
            plt.savefig(
                'pictures/'
                + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                + '-ponds_eval.png', dpi=150
            )
        if show_pic:
            plt.show()
