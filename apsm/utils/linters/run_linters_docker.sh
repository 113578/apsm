#!/bin/bash

DOCKER_FILE=$1

echo "Проверка hadolint..."
"$DOCKER_FILE" > docker run --rm -i hadolint/hadolint

echo "Код проверен."
