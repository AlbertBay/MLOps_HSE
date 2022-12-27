from flask import Flask
from flask_restx import Api, Resource, reqparse
import json
import os
from api_logic.utils import fit_save, predict_log, get_trained, delete


app = Flask(__name__)
api = Api(app)

@api.route('/all_available_models',
           methods=['GET'],
           doc={'description': 'Get available models'})
class Available_Models(Resource):
    @api.response(200, 'OK')
    @api.response(500, 'INTERNAL SERVER ERROR')
    def get(self):
        return 'Models available to train: RandomForestClassifier, DT', 200


@api.route('/all_trained_models',
           methods=['GET'],
           doc={'description': 'Get information about learned models'})
class Trained_Models(Resource):
    @api.response(200, 'OK')
    @api.response(500, 'INTERNAL SERVER ERROR')
    def get(self):
        #with open("param_fitted_models.json", "r") as jsonFile:
        #    trained_models = json.load(jsonFile)
        #if trained_models == {}:
        #    return 'No available trained models', 200
        #else:
        #    return trained_models, 200
        return get_trained()


params_of_training = reqparse.RequestParser()
params_of_training.add_argument('id',
                                type=str,
                                help='Unique id',
                                required=True)
params_of_training.add_argument('type',
                                type=str,
                                help='Type of the model',
                                choices=['DT', 'RandomForestClassifier'],
                                required=True)
params_of_training.add_argument('params',
                                type=str,
                                help='parameters for the model {"n_estimators": [100]}')
params_of_training.add_argument('target',
                                type=str,
                                help='name of the target column (turnover.csv default)')
params_of_training.add_argument('path',
                                type=str,
                                help='data path (turnover.csv default)')
@api.route('/fit_model',
           methods=['POST'],
           doc={'description': 'Learn and dump model'})
class Fit_Model(Resource):
    @api.expect(params_of_training)
    @api.response(200, 'OK')
    @api.response(400, 'BAD REQUEST')
    @api.response(500, 'INTERNAL SERVER ERROR')
    def post(self):
        args = params_of_training.parse_args()
        return fit_save(args)


# fields for inference
params_of_inference = reqparse.RequestParser()
params_of_inference.add_argument('id', help='Model"s ID for predition', required=True)
params_of_inference.add_argument('target',
                                 type=str,
                                 help='name of the target column (turnover.csv default)')
params_of_inference.add_argument('path',
                                 type=str,
                                 help='data path (turnover.csv default)')
@api.route('/predict',
           methods=['GET'],
           doc={'description': 'Make prediction'})
class Predict(Resource):
    @api.expect(params_of_inference)
    @api.response(200, 'OK')
    @api.response(400, 'BAD REQUEST')
    @api.response(500, 'INTERNAL SERVER ERROR')
    def get(self):
        args = params_of_inference.parse_args()
        return predict_log(args)


# fields for deleting
params_of_deleting = reqparse.RequestParser()
params_of_deleting.add_argument('id', help='Models"s ID for deleting', required=True)
@api.route('/delete_model')
class Delete_Model(Resource):
    @api.expect(params_of_deleting)
    @api.response(200, 'OK')
    @api.response(400, 'BAD REQUEST')
    @api.response(500, 'INTERNAL SERVER ERROR')
    def delete(self):
        args = params_of_deleting.parse_args()
        id_model = args.id
        if id_model + '.pkl' not in os.listdir('fitted_models/'):
            return f'Trained model with ID = {id_model} doesnt exist', 400
        else:
            delete(id_model)
            return f'Model with ID = {id_model} is deleted', 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
