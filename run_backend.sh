# if conda command is not recognized, please reopen shell
# change directory to image2geolocation


cd app
pip install -r requirements.txt


FILE="$(pwd)/backend_models/model_checkpoints/model_efficientnetv2_rw_m_22.pt"
if [[ ! -f "$FILE" ]]
then
    cd backend_models
    mkdir model_checkpoints
    cd model_checkpoints
    pip install gdown
    gdown https://drive.google.com/uc?id=1l_txSXtW9__A89scB56WpIB89NBVSTnd
    cd ..
    cd ..
fi

python3 manage.py runserver 0.0.0.0:8000
