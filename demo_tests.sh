#!/bin/bash -e


pushd ./examples
echo "cleaning old compiled notebooks"
rm -f *.py

echo "Converting notebooks"
jupyter nbconvert --to script *.ipynb

echo "Running notebooks"
python *.py

popd
