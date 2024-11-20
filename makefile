build:
	python3 -m venv venv
	venv/bin/pip3 install -r requirements.txt

cornell_box_ex:
	venv/bin/python3 cornell_box.py