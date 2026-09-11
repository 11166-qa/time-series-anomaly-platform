import sqlite3
import pandas as pd


class SQLiteManager:

    def __init__(self, db_path="results.db"):
        self.conn = sqlite3.connect(db_path)


    def create_table(self):

        sql = """
        CREATE TABLE IF NOT EXISTS prediction_results(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            model TEXT,
            actual REAL,
            prediction REAL,
            residual REAL
        )
        """

        self.conn.execute(sql)
        self.conn.commit()



    def save_predictions(self, df):

        df.to_sql(
            "prediction_results",
            self.conn,
            if_exists="append",
            index=False
        )



    def query_model(self, model):

        sql = """
        SELECT *
        FROM prediction_results
        WHERE model=?
        """

        return pd.read_sql(
            sql,
            self.conn,
            params=(model,)
        )
