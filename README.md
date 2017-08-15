# BioAsqLG

The webservices code is written in Flask.
The main code is basic.py.
Command to run: python basic.py ../input/BioASQ-trainingDataset5b.json.
The web interface opens in local host at http://127.0.0.1:5000/.
There are about 18 different configurations from 3 3 modules of our system.
Currently the code has clustering algorithm with 3 choices of No expansion/ UMLS expansion/ SNOMEDCT expansion.
Will add the MMR code with soft and hard constraints.
System requirements include: pymedtermino, nltk, sklearn, flask.
