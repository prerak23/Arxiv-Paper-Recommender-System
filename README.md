# Arxiv-Paper-Recommender-System

## These Libraries Should Be Installed Before Running This Service
1. Numpy
2. Pandas 
3. Scikit-learn
4. Spacy 
5. Tensorflow By Using This Command (pip3 install --quiet "tensorflow>=1.7")
6. Tensorflow_hub By Using This Command (pip3 install --quiet tensorflow-hub)
7. Universal Sentance Encoder
8. _Input text should only contain text and should not contain latex formulated formulas for better results_
9. All the files including the file named as _db_encoded.npy_ should be in the same dir/folders 

## Introduction
This is a research paper recommender system which matches similarity between papers on the basis of semantic similarity of the abstract of the paper 
It was a part of my research project during my summer internship at Loria , Nancy , France

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

### Basically there are 7 important files 
1. **service.py** :-
This python file contains code that clean and encode your input abstract and compare it with the diffrent already encoded abstracts present in the database and at the end gives you the 10 most similar papers by using **cosine similarity** as a metric to detrimine similarity scores between the papers the service is quite fast as we have used **a diffrent approach** to compute **cosine similarity** and did some tricks on numpy arrays to make everything super fast and efficient eventhough the database is 1.5 million papers it usually take about **50 seconds** to recomend us the 10 most similar papers. 

2. **extract_data.py** :- 
If you want to make your own database from scratch then run this python file which will create a **database.csv** file at the end of the programe which will contain theses information **id , title , abstract , categories**  
To change the time period of the data to be extracted you can change on this line in the file as follows
```
ax = Scraper(category=xp, date_from='**yyyy-mm-dd**',date_until='2019-08-01')

```
3. **database_clean_and_encode.py** :-
After running the above mentioned file you have to run this python file which will try to clean up your abstract which is present in the file named as **database.csv**  using the python program named as **clean_abs_stat.py** and will also encode the abstract using python programe named as **tfs_encode.py** during its running it will try to create two files in the mean-time 
**a) database_clean.csv** (This file is similar to database.csv but the abstract present in this file are cleaned)
**b) db_encoded.npy**     (This Numpy file is the final result of this program which will consist the encoded version of your abstract)

```

dir_path=os.getcwd()
os_specific='/' if 'posix' in os.name else '\\'
db=pd.read_csv(dir_path+os_specific+"database.csv")
final_path=dir_path+os_specific+"database.csv"
clean_abs_stat.clean_abs(final_path).to_csv("database_clean.csv")
tfs_encode("database_clean.csv",False)

```
4. **update_database.py** :-
This python file updates the database according to the last updated date on which the file **db_encoded.npy** was updated or created 
while running this file one more file is created in the meantime named as **updated_data** to store the updated data that is gathered from the oaipmh requests
then we clean the abstract in this temprory file and encode it after that we then basically append this new encoded data to the file **db_encoded.npy**

```
updated_data=pd.DataFrame([])
for xp in ll_sets:
    ax = Scraper(category=xp, date_from=date_load ,date_until=datetime.date.today())
    output=ax.scrape()
    df=pd.DataFrame(output,columns=['id', 'title', 'categories', 'abstract', 'doi', 'created', 'updated', 'authors','url'])
    df=df.drop(columns=['doi','authors','created','updated','url'])
    print(len(df))
    updated_data=pd.concat([updated_data,df],sort=False,ignore_index=True)


updated_data.to_csv('updated_data.csv')
dir_path=os.getcwd()
os_specific="/" if "posix" in os.name else "\\"
clean_abs_stat.clean_abs(dir_path+os_specific+'updated_data.csv').to_csv('updated_data.csv')
tfs_encode.encode('updated_data.csv',True)

```
5. **clean_abs_stat.py** :-
This python file is used to clean the abstract in the database file or in the updated basically to clean the abstract I have created a 4 stage pipeline for that which is described as follows
**a) Pipeline 1** :- As abstract of the research papers contains various math symbols so our first stage would be to remove them hence we convert every $ $ to an $ * because as per latex every formula is between two $ $ signs so we convert the other $ to an * so it will become $ * to denote its a math formula
**b) Pipeline 2** :-  Removing the new-line charachters
**c) Pipeline 3** :- Removing the whole math symbols from the abstracts
**d) Pipeline 4** :- Remove the symbols except from the strong punctations marks from the abstract

6. **tfs_encode.py**
 This python program encodes the abstract present in the database and at the end create a file called as **db_encoded.npy** 
 the method in the code excpects two parameters one is _update_ (Boolean) if its an update which is expected to take place or not and a
 file_name which is only needed when there is an update which needs to be done on the pre-defined database
7. **drop_duplicate.py**
   This python file is used to drop the duplicates from the numpy database in a fast way during the extraction of data from the oiapmh 
   request there were a lot of papers which were published twice or thrice because of some reasons on arxiv so when you do an oaipmh 
   request it usually give duplicate data entries. Hence the task of this python script is to remove those duplicate data entries using
   _title_ as the subject to find the duplicates in the encoded data file known as _db_encoded.npy_ 

## Analysis of the encodings produced by universal sentance encoder

Due to lack of gold labelled dataset for this specific task we struggled to provide an evalutaion of the encodings produced for these abstract by the universal sentence encoder 
But we still tried to do the following things to evaluate encodings in our dataset 

### Spherical K means clustering 

I tried to do clustering with the help of spehrical k means clustering because the dimension produced by the encoded embedding per abstract were in 512 dimensions and thus it is difficult to use any other clustering methods which uses l2 measure manhattant distance or euclidean distance to define clusters.
I have tried various approaches and algorithms by sci-kit learn for clustering but the results were not satisfying as every time I used DBSCAN, Agglomerative clustering or simple k means clustering it always give me out an degenerative clusters 

I have tried with various parameters as well like increasing the clusters and other parameters which are relevant to the type of algorithm that i used it was not working out properly

The package that i used for this clustering task by jasonalaska use the standard scikit learn k -means clustering but modified it a bit to work out for calculating spherical k means clusters as well

**Plot cluster points**
I have also used TSNE to get out the clustering but found it its just an algorithm to minimize dimensionality of the data so that you can plot it on the 2-d or 3-d surfaces

High-dimensional datasets can be very difficult to visualize. While data in two or three dimensions can be plotted to show the inherent structure of the data, equivalent high-dimensional plots are much less intuitive. To aid visualization of the structure of a dataset, the dimension must be reduced in some way.

**Metrics on the result of k-means clustering**
I have used silhouette metric because for this task i did not have labels to the train system and this and the package i used for spherical k means clustering uses this metric to define how good their cluster are in actual

I found out by experimenting with parameters that less the clusters the more silhouette score i was getting which means better the clusters formed by the package

**Why I havent used the labels as categories/sets are already defined with each abstract ?**
The clusters that are formed by spehrical k means clustering cannot be evaluated because of inavaliabilty of gold dataset for this specific task to compare our results.
Because we are clustering on the basis of semantic similarity of every abstract the problem arises becuase in a cluster you can have papers which have similar semantic similarity but belongs to an diffirent categories/set alltogether 
For Example :- Suppose Cluster 0 have 75 % of the paper which belong to the category _cs_ and rest 25% of papers belong to the diffrent sets like _physics or maths_ these 25 % paper are somewhat related or you can say have same semantic similarity as the 75 % of the paper present in the same cluster so , now the question comes what label or how would you define this cluster.

The best thing for evaluation is to go and read each paper and then try define the labels per cluster but that is tedius task on its own 
if there would be an avaliabilty of gold dataset to compare then the evaluation would be much easier to do 


**Important Observations**
1. Some papers which belong to same set and also belong to same category sometime have a lot of dissimilarity between them

2. Sometimes the abstract contains more formulas then text hence with less words in general it is difficult to extract out the features of that extract















