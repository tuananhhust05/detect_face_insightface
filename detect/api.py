import json
from flask import Flask, jsonify, request
from flask_cors import CORS
app = Flask(__name__,static_url_path='', 
            static_folder='/home/poc4a5000/storage_facesx')
CORS(app)
# from detect.detect import handle_main

@app.route('/analyst', methods=['GET'])
def get_employees():
    case_id = request.args.get('case_id')
    tracking_folder = request.args.get('tracking_folder')
    target_folder = request.args.get('target_folder')
    # def handle_main(case_id, tracking_folder, target_folder)
    print("target_folder",target_folder,tracking_folder,case_id)
    return jsonify({
        "data":"ok"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5237, host='0.0.0.0')