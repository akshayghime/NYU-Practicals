"""
Simple stand-alone script showing end-to-end training of a regression model using Metaflow.
"""

from metaflow import FlowSpec, step, Parameter, IncludeFile, current
from comet_ml import Experiment
from datetime import datetime
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# make sure we are running locally for this
assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'

class FraudDetectionFlow(FlowSpec):
    """
    Logistic regression flow is a minimal DAG showcasing reading data from a file
    and training the model successfully
    """

    #exp = Experiment(api_key="HDwtDordk4T6HcaKB3ZlxLTYn",
    #                 project_name="fraud_detection",
    #                auto_param_logging=False)
    @step
    def start(self):
        """
        Start up and print out some info to make sure evrything is okay on metaflow side
        """
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" %current.username)
        self.next(self.read_data)
    
    @step
    def read_data(self):
        """
        Read all datasets
        """
        self.df_payments = pd.read_csv("payments.csv")
        self.df_merchants = pd.read_csv("merchants.csv")
        self.df_buyers = pd.read_csv("buyers.csv")
        
        # go to the next step
        self.next(self.feature_engineering)
    
    @step
    def feature_engineering(self):
        """
        Feature engineering for before training
        """
        df = self.df_payments.copy()
        df['is_fraud'] = df['chargeback_timestamp'].notna().astype(int)

        # Merge with merchants and buyers
        merged = pd.merge(df, self.df_merchants, left_on='merchant_id', right_on='id', how='left')
        merged = pd.merge(merged, self.df_buyers, left_on='buyer_id', right_on='id', how='left')
        
        if 'id' in merged.columns:
            merged.drop(columns=['id'], inplace=True)

        merged = merged.rename(columns={
            'country_x': 'merchant_country',
            'country_y': 'buyer_country',
            'category': 'merchant_category'
        })

        merged = merged.drop(columns=['id_x', 'id_y'])
        merged['transaction_timestamp'] = pd.to_datetime(merged['transaction_timestamp'])
        merged_sorted = merged.sort_values('transaction_timestamp').reset_index(drop=True)

        # Fraud rates
        merged_sorted['merchant_fraud_rate'] = (
            merged_sorted.groupby('merchant_id')['is_fraud']
            .expanding().mean().shift(1).reset_index(level=0, drop=True).fillna(0)
        )
        merged_sorted['buyer_fraud_rate'] = (
            merged_sorted.groupby('buyer_id')['is_fraud']
            .expanding().mean().shift(1).reset_index(level=0, drop=True).fillna(0)
        )

        merged_sorted["transaction_date"] = merged_sorted["transaction_timestamp"].dt.date

        # One-hot encoding
        self.df_encoded = pd.get_dummies(merged_sorted, columns=[
            "merchant_category", "merchant_country", "buyer_country"
        ])
        self.next(self.split_data)

    @step
    def split_data(self):
        L_DAYS = 32
        TEST_WINDOW_DAYS = 30

        df = self.df_encoded.copy()
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
        most_recent = df["transaction_date"].max()

        test_end = most_recent - pd.Timedelta(days=L_DAYS)
        test_start = test_end - pd.Timedelta(days=TEST_WINDOW_DAYS)
        train_end = test_start - pd.Timedelta(days=L_DAYS)

        train_df = df[df["transaction_date"] <= train_end]
        test_df = df[(df["transaction_date"] >= test_start) & (df["transaction_date"] <= test_end)]

        drop_cols = ['buyer_id','transaction_timestamp','chargeback_timestamp',
                     'is_fraud','merchant_id','transaction_date']

        self.X_train = train_df.drop(columns=drop_cols)
        self.y_train = train_df["is_fraud"]
        self.X_test = test_df.drop(columns=drop_cols)
        self.y_test = test_df["is_fraud"]

        self.next(self.scale_data)
    
    @step
    def scale_data(self):
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        self.next(self.train_lr,self.train_rf)

    @step
    def train_lr(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        
        # Log to Comet
        exp = Experiment(api_key="HDwtDordk4T6HcaKB3ZlxLTYn",
                   project_name="fraud_detection",
                   auto_param_logging=False)
        #exp.log_parameter("max_iter",max_iter=1000)

        # Train the model
        model = LogisticRegression(class_weight="balanced", max_iter=1000)
        model.fit(self.X_train_scaled, self.y_train)
        
         # Predict and calculate AUC
        y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
        auc = roc_auc_score(self.y_test, y_pred_proba)

        print(f"Logistic Regression AUC: {auc:.4f}")

        # Log metrics
        exp.log_metric("auc", auc)

        # Store results for join step
        self.model = model
        self.auc = auc
        self.y_pred_proba = y_pred_proba

        self.next(self.join)

    @step
    def train_rf(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score

        exp = Experiment(api_key="HDwtDordk4T6HcaKB3ZlxLTYn",
                   project_name="fraud_detection",
                   auto_param_logging=False)
        #exp.log_parameter("n_estimators", n_estimators=100)

        # Train the model
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42
        )
        model.fit(self.X_train_scaled, self.y_train)

        # Predict and calculate AUC
        y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
        auc = roc_auc_score(self.y_test, y_pred_proba)

        print(f"Random Forest AUC: {auc:.4f}")

        # Log metrics
        exp.log_metric("auc", auc)

        # Store results for later use
        self.model = model
        self.auc = auc
        self.y_pred_proba = y_pred_proba

        self.next(self.join)

    @step
    def join(self, inputs):
        best = max(inputs, key=lambda x: x.auc)
        self.best_model = best.model
        self.best_auc = best.auc
        print(f"Best model with auc={self.best_auc:.4f}")
        self.next(self.end)


    @step
    def end(self):
        print("All done ")

if __name__ == '__main__':
    FraudDetectionFlow()