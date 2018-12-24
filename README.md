# ml-app-model
General model to deliver ml apps

### Add to gitignore
#### data
*.csv
*.xlsx
*.xls
*.pk
*.pkl
data/
models/*.pk
models/*.pkl


### Setup venv
alias cenv='virtualenv -p python3 .venv'
alias senv='source .venv/bin/activate'
alias denv='deactivate .venv/bin/activate'

### Install
pip3 install pandas sklearn Flask requests dill