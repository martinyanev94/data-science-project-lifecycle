# Initialize a DVC repository
dvc init

# Track our dataset
dvc add data/dataset.csv

# Stage the changes in Git
git add data/dataset.csv.dvc .gitignore
git commit -m "Add dataset"

# Track your model
dvc run -n train_model -d src/train.py -d data/dataset.csv -o model/model.pkl python src/train.py
