from models.Pipe import loader, Pipe
import json
import os
import pickle
from sqlalchemy import create_engine
import pandas as pd

db_name = 'database'
db_user = 'username'
db_pass = 'secret'
db_host = 'dbase'
db_port = '5000'

db_string = 'postgresql://{}:{}@{}:{}/{}'.format(db_user, db_pass, db_host, db_port, db_name)
db = create_engine(db_string)

def delete(id_model):
    db.execution_options(autocommit=True).execute(
        f"""
                       DELETE
                       FROM mdls
                       WHERE "id" = '{id_model}';
                       """
    )
    db.dispose()
    os.remove('fitted_models/' + str(id_model) + '.pkl')


def exist_id():
    __idList = pd.read_sql_query(
        """
        SELECT DISTINCT id
        FROM mdls;
        """,
        db
    ).id.tolist()
    db.dispose()

    return __idList


def get_trained():
    __trained = pd.read_sql_query(
        """
        SELECT *
        FROM mdls;
        """,
        db
    ).to_dict(orient="index")
    db.dispose()

    return __trained


def fit_save(args):
    id_model = args.id
    type_model = args.type
    data_path = 'data/turnover.csv' if args.path is None else args.path
    target_column = 'event' if args.target is None else args.target
    model_params = {} if args.params is None else json.loads(args.params.replace("'", "\""))
    if id_model + '.pkl' in os.listdir('fitted_models/'):
        return 'Model with this id is already exist, change different id', 400
    else:
        data, target = loader(data_path, target_column)
        pipe = Pipe(id_model, type_model)
        flag, par = pipe.fit(data, target, model_params)
        if flag == 0:
            return f'Parameter {par} of {type_model} out of naming', 400
        else:
            # сохраняю все параметры обученной модели
            with open(f'fitted_models/{id_model}.pkl', 'wb+') as file:
                pickle.dump(pipe, file)
            db.execution_options(autocommit=True).execute(
                f"""
                              INSERT INTO mdls ("id", "params")
                              VALUES (%s,%s,%s);
                              """,
                (id_model, model_params)
            )
            db.dispose()
            #with open("param_fitted_models.json", "r") as jsonFile:
            #    data = json.load(jsonFile)
            #data[id_model] = {'name_model': type_model, 'model_params': model_params}
            #with open("param_fitted_models.json", "w") as jsonFile:
            #    json.dump(data, jsonFile)
            return 'Model trained', 200


def predict_log(args):
    id_model = args.id
    data_path = 'data/turnover.csv' if args.path is None else args.path
    target_column = 'event' if args.target is None else args.target
    if id_model + '.pkl' not in os.listdir('fitted_models/'):
        return f'No trained model with ID {id_model}', 400
    else:
        with open('fitted_models/' + str(id_model) + '.pkl', 'rb') as file:
            pipe = pickle.load(file)
        data, target = loader(data_path, target_column)
        y_pred = pipe.predict(data)
        return json.dumps({f'ID = {id_model}': list(y_pred)})
