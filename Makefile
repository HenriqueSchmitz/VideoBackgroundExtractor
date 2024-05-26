compile:
	rm -rf dist build
	pip wheel -w dist --no-deps .

test-deploy:
	twine upload --repository testpypi dist/*