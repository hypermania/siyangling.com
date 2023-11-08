#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Copying CV to ${SCRIPT_DIR}/files"
cp ~/Dropbox/Files/Materials/CV/main.pdf ${SCRIPT_DIR}/files/CV.pdf
