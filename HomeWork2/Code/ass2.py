import pandas as pd

from sklearn.preprocessing import MinMaxScaler


class TitanicSinkingModel:
    def __init__(self, filename):
        self.df = pd.read_csv(filename)
        self.df_normalized = None

    def preprocessing(self):
        scaler = MinMaxScaler()
        self.df_normalized = pd.DataFrame(scaler.fit_transform(self.df.values),
                                          columns=self.df.columns, index=self.df.index)

    def print_df_normalized(self):
        print(self.df_normalized.head().to_string())


if __name__ == '__main__':
    titianic_sinking_model = TitanicSinkingModel("titanic.csv")
    titianic_sinking_model.preprocessing()
    titianic_sinking_model.print_df_normalized()
