cd ../src
pdoc --html . --output-dir ../docs
cp -r ../docs/src/* ../docs/
rm -rf ../docs/src
