install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

train:
	python train.py