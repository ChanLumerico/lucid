cd devtools/scripts
./black.sh ../../lucid
./rm_pyc.sh ../../lucid
python --version
cd ../../docs
# rm -r build/
# make html
make html