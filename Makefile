setup : $./requirement.txt
	@python3 -m venv venv
	@./venv/bin/activate
	@pip install -r requirement
	@deactivate
	@echo "SETUP COMPLETED"

main : 
	@./venv/bin/activate
	@python3 main.py
	@deactivate

