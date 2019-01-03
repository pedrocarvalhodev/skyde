## Skyde : an ml-web-app
General model to deliver ml apps

## Run Model
1. gunicorn --workers=2 --timeout 600 --bind 0.0.0.0:8000 app:app
2. http://0.0.0.0:8000

## Install
pip3 install pandas sklearn Flask requests dill