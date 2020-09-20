cd ../src
pdoc --html . --output-dir ../docs
cp -r ../docs/src/* ../docs/
rm -rf ../docs/src
cd ../docs
python style_docs.py

# build readme for nbs
jupytext --to md ../notebooks/*.py
jupytext --to md ../notebooks/*.ipynb
python process_nbs.py

