#!/bin/bash -l
set -e
if [ "$#" -eq 0 ]; then
  # Kill cache, pytest complains about it if running local and docker tests in mapped volume
  find tests  -type d -name '__pycache__' -print0 | xargs -0 rm -rf {}
  # Make sure the service itself is installed
  poetry install
  # Make sure pre-commit checks were not missed and run tests
  git config --global --add safe.directory /app
  poetry run pre-commit install
  pre-commit run --all-files
  pytest -v --junitxml=pytest.xml tests/
else
  exec "$@"
fi
