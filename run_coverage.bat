coverage run --omit=test_algorithms.py  -m unittest discover
coverage report -i
coverage html -i