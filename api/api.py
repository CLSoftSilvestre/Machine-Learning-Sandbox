# -*- coding: utf-8 -*-
"""
Created on 09/07/2024

@author: CSilvestre

"""

from flask import Flask
from flask_restx import Api, Resource

app = Flask(__name__, instance_relative_config=True)
app.config['RESTPLUS_MASK_SWAGGER'] = False
api = Api(app)


@api.route('/teste')
class teste(Resource):
    def get(self):
        return 'hello'

# On going...

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5003, debug=True)