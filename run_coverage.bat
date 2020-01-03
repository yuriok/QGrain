coverage run --source=. --omit=*/test_*.py  -m unittest discover ./tests
coverage report -i
coverage html -i