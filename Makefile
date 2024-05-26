build:
	rm -rf dist build
	pip wheel -w dist --no-deps .
	rm -rf build

test-deploy:
	twine upload --repository testpypi dist/*