#!/bin/bash -e


pushd ./examples
echo "cleaning old compiled notebooks"
rm -f *.py

echo "Converting notebooks"
jupyter nbconvert --to script *.ipynb

echo "Running notebooks"
for f in *.py;
do
echo "Running $f";
ipython $f;
done

popd
