import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_curve, auc


# import numpy as np
# import seaborn as sns
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

    def find_optimal_decision_tree(self):
        # min_samples_leaf the minmum number of leaves a split
        # can happen according to the value of entropy
        auc_tain = {}
        auc_test = {}
        for i in range(2, 21):
            descion_tree = tree.DecisionTreeClassifier(min_samples_leaf=i)
            descion_tree.fit(self.data_train, self.target_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(self.target_train, descion_tree.predict(self.data_train))
            auc_tain[i] = auc(false_positive_rate, true_positive_rate)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(self.target_test, descion_tree.predict(self.data_test))
            auc_test[i] = auc(false_positive_rate, true_positive_rate)
            # fig = plt.figure()
            # d = {'min_samples_leaf': np.array(list(auc_tain)), 'AUC score': np.array(list(auc_tain.values()))}
            # pd_plot = pd.DataFrame(d)
            # sns.lineplot(x='min_samples_leaf', y='AUC score', data=pd_plot)
            # plt.show()
            # fig.savefig('plot_train.png')
        self.clf = tree.DecisionTreeClassifier(min_samples_leaf=6)
        self.clf.fit(self.data_train, self.target_train)
        # dot_data = tree.export_graphviz(self.clf, out_file=None)
        # graph = graphviz.Source(dot_data)
        # graph.format = 'jpg'
        # graph.render("decision_tree_plot_optimal")


def print_df_normalized(self):
    print(self.df_normalized.head().to_string())


if __name__ == '__main__':
    titianic_sinking_model = TitanicSinkingModel("titanic.csv")
    titianic_sinking_model.preprocessing()
    # titianic_sinking_model.print_df_normalized()
    # print(titianic_sinking_model.data_train.to_string())
    # titianic_sinking_model.fit_decision_tree()
    titianic_sinking_model.find_optimal_decision_tree()
