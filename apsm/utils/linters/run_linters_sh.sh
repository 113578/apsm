#!/bin/bash

SHELL_SCRIPT=$1

echo "Проверка ShellCheck..."
docker run --rm -v "$(pwd):/mnt" koalaman/shellcheck:stable "/mnt/$SHELL_SCRIPT"

echo "Код проверен."
