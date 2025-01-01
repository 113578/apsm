#!/bin/bash

SCRIPT_FILE=$1

echo "Проверка pycodestyle..."
pycodestyle "$SCRIPT_FILE" --max-line-length=120

echo "Проверка flake8..."
flake8 "$SCRIPT_FILE" --max-line-length=120

echo "Проверка pylint..."
pylint "$SCRIPT_FILE" --max-line-length=120 --disable="C0103,C0114,C0115"

echo "Код проверен."
