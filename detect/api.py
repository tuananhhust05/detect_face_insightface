import json
from flask import Flask, jsonify, request
app = Flask(__name__,static_url_path='', 
            static_folder='')


@app.route('/analyst', methods=['GET'])
def get_employees():
    case_id = request.args.get('case_id')
    tracking_folder = request.args.get('tracking_folder')
    target_folder = request.args.get('target_folder')
    print(target_folder)
    return jsonify({
        "data":"ok"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5234, host='0.0.0.0')