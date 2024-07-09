
from flask import Blueprint, request

blueprint = Blueprint('api', __name__, url_prefix='/basic_api')

@blueprint.route('/hello_world')
def hello_world():
    return {'message': 'Hello World!'}

@blueprint.route('/entities', methods=['GET', 'POST'])
def entities():
    if request.method == 'GET':
        return {
            'message': 'This endpoint should return a list of entities',
            'method': request.method
        }
    
    if request.method == 'POST':
        return {
            'message': 'This endpoint should create an entity',
            'method': request.method,
            'body':request.json
        }

