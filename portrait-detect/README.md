This is for DB expansion for person task of Gradient SN.
Extract shoulder and head from person images.
It will be used for generation of images.

Generated portrait will be saved in "./detected_faces" folder.

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run.py