pip install dvc
dvc init
dvc add data/dataset.csv
git add data/dataset.csv.dvc .gitignore
git commit -m "Add dataset with DVC"
