# -*- coding: utf-8 -*-
"""
Created on 09/07/2024

@author: CSilvestre
"""

from flask import Flask
from blueprints.basic_endpoints import blueprint as basic_endpoints

app = Flask(__name__, instance_relative_config=True)
app.register_blueprint(basic_endpoints)

# On going...

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5003, debug=True)