setup:
	python -m venv venv
	./venv/bin/pip install -r requirements.txt

clean:
	rm -f data/*