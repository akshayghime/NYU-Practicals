install:
	#install commands
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	#format code
	black *.py