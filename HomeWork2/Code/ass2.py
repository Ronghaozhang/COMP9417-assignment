import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# import graphviz
# import matplotlib.pyplot as plt

class TitanicSinkingModel:
    def __init__(self, filename):
        self.df = pd.read_csv(filename)
        self.df_normalized = None
        self.data_train = None
        self.data_test = None
        self.target_train = None
        self.target_test = None
        self.clf = None

    def preprocessing(self):
        # this can be done because the target value is either 0 or 1
        # thus apply the normalization have no effect on them
        scaler = MinMaxScaler()
        self.df_normalized = pd.DataFrame(scaler.fit_transform(self.df.values),
                                          columns=self.df.columns, index=self.df.index)
        self.split_dataframe()

    def split_dataframe(self):
        data = self.df_normalized[
            [col for col in self.df_normalized.columns if col != 'Survived']]
        target = self.df_normalized['Survived']
        # split the test and train data set
        self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(data, target,
                                                                                                test_size=0.3,
                                                                                                shuffle=False)

    def fit_decision_tree(self):
        self.clf = tree.DecisionTreeClassifier()
        self.clf = self.clf.fit(self.data_train, self.target_train)

        # dot_data = tree.export_graphviz(clf, out_file=None)
        # graph = graphviz.Source(dot_data)
        # graph.format = 'jpg'
        # graph.render("decision_tree_plot_orignial")
        labels_test = self.clf.predict(self.data_test)
        acc = accuracy_score(labels_test, self.target_test)
        print("acc for test set is : " + str(acc))
        labels_test2 = self.clf.predict(self.data_train)
        acc2 = accuracy_score(labels_test2, self.target_train)
        print("acc for train set is : " + str(acc2))


def print_df_normalized(self):
    print(self.df_normalized.head().to_string())


if __name__ == '__main__':
    titianic_sinking_model = TitanicSinkingModel("titanic.csv")
    titianic_sinking_model.preprocessing()
    # titianic_sinking_model.print_df_normalized()
    # print(titianic_sinking_model.data_train.to_string())
    titianic_sinking_model.fit_decision_tree()
