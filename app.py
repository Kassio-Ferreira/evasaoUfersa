from flask import Flask
from flask import request

import utils as ut

app = Flask(__name__)


@app.route('/', methods=['POST'])
def abandono():
    content = request.get_json()
    objeto = ut.formata_objeto(content)
    return ut.seletor(content, objeto)


if __name__ == '__main__':
    app.run(debug=True)
