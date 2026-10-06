create-env:
	python3 -m venv venv

install-packets:
	./venv/bin/pip install -r requirements.txt

upload-database:
	./venv/bin/python src/database/upload_csv.py

train-model:
	./venv/bin/python src/machine_learning/train_random_forest.py
