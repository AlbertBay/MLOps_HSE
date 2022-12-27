import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GridSearchCV


MODELS = {'DT':  DecisionTreeClassifier(), 'RandomForestClassifier':  RandomForestClassifier() }


def loader(path: str, target_col: str):
    """
    Подготовка данных для обучения и предсказания.
    Данные представляют собой датафрейм с признаками машин.

    Param:
    file_name: str

    Return:
    x: pd.DataFrame
    y: pd.Series

    """
    df = pd.read_csv(path)
    target = df[target_col]
    data = df.drop([target_col], axis=1)
    return data, target


class Preprocessing:
    cat: list
    column_transformer: ColumnTransformer
    fill_values: dict
    is_linear: bool
    num: list

    def __init__(self, cat: list, num: list):
        """

        @param cat:
        @param num:
        """
        self.cat = cat
        self.num = num
        self.column_transformer = ColumnTransformer([
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), self.cat),
            ('chill', 'passthrough', self.num)
        ])

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """

        @param df:
        @return:
        """
        df[self.cat] = df[self.cat].astype(str)
        df[self.num] = df[self.num].astype(float)
        df[self.num] = df[self.num].fillna(value=-1000)
        df[self.cat] = df[self.cat].fillna(-1000)
        df = self.column_transformer.fit_transform(df)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """

        @param df:
        @return:
        """
        df[self.cat] = df[self.cat].astype(str)
        df[self.num] = df[self.num].astype(float)
        df = self.column_transformer.transform(df)
        return df


class Pipe:
    def __init__(self, id: str, model: str):
        """
        @param id:
        @param model:
        """
        self.id = id
        self.model = model
        self.prep = None
        self.final_model = None

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series,
            params: dict, cv: int = 5):
        """

        @param x_train:
        @param y_train:
        @param params:
        @param cv:
        """
        for param in params:
            if param not in MODELS[self.model].get_params().keys():
                return 0, param
        cat = x_train.select_dtypes(['object']).columns.tolist()
        num = x_train.select_dtypes(['int64', 'float64']).columns.tolist()
        self.prep = Preprocessing(cat=cat, num=num)
        x_train = self.prep.fit_transform(x_train)
        self.final_model = GridSearchCV(MODELS[self.model],
                                   params,
                                   cv=cv,
                                   scoring='roc_auc',
                                   return_train_score=False,
                                   verbose=0,
                                   error_score='raise')
        self.final_model.fit(x_train, y_train)
        return 1, 'ok'

    def predict(self, x_test: pd.DataFrame):
        """

        @param x_test:
        """

        x_test = self.prep.transform(x_test)
        y_pred_proba = self.final_model.predict_proba(x_test)[:, 1]

        return y_pred_proba
