# Arxiv-Paper-Recommender-System-

## These Libraries Should Be Installed Before Running This Service
1. Numpy
2. Pandas 
3. Scikit-learn
4. Spacy
5. Tensorflow
6. Tensorflow_hub


,clean_abs_stat.py,tfs_encode.py
## Introduction
This is a research paper recommender system which matches similarity between papers on the basis of semantic similarity of the abstract of the paper 

## How To Use This Service 
1. Clone the github repo 
2. Download the database (db_encoded.npy) which consist of already encoded abstracts of 1.5 million research paper from arxiv 
3. Then run the python file 
4. Input the abstract of your research paper that you want to find similarity with 
5. As a result you will have the 10 most similar research paper that have the same semantic meaning in their abstract as your input abstract
## How to Update The Database
1. Just run the file update_database.py 
2. The current database on your system will get updated with the new database from arxiv via oipamh request
3. Depending on when the database was last updated this updating can take several minuites to sevral hours to update
4. **So, if you use this service too frequent it is best to update the database every month**
## Info what goes on behind the scences
### This application works on an already trained sentance encoder model called as Universal Sentance Encoder this model runs on DAN (Deep Averaging Network ) architecture the model is trained using multi-task learning 

### Basically there are 5 important files 
1. **service.py** :-
This python file contains code that clean and encode your input abstract and compare it with the diffrent already encoded abstracts present in the database and at the end gives you the 10 most similar papers by using **cosine similarity** as a metric to detrimine similarity scores between the papers as of now it using **_brute-force technique_** to give 10 most similar papers but we can obviously make it more faster to run
2. **extract_data.py** :- 
If you want to make your own database of from the scratch then run this python file which will create a **database.csv** file at the end which will contain information these columns of information **id , title , abstract , categories** 
To change the time period of the data to be extracted you can change on this line in the file as follows
```
ax = Scraper(category=xp, date_from='**yyyy-mm-dd**',date_until='2019-08-01')

```
3. **database_clean_and_encode.py** :-
After running the above mentioned file you have to run this python file which will try to clean up your abstract which is present in the file named as **database.csv**  using the python program named as **clean_abs_stat.py** and will also encode the abstract using python programe named as **tfs_encode.py** during its running it will try to create two files in the mean-time 
a) **database_clean.csv** (This file is similar to database.csv but the abstract present in this file are cleaned)
b) **db_encoded.npy**     (This Numpy file is the final result of this program which will consist the encoded version of your abstract)

```

dir_path=os.getcwd()
os_specific='/' if 'posix' in os.name else '\\'
db=pd.read_csv(dir_path+os_specific+"database.csv")
final_path=dir_path+os_specific+"database.csv"
clean_abs_stat.clean_abs(final_path).to_csv("database_clean.csv")
tfs_encode("database_clean.csv",False)

```
4. **update_database.py** :-









