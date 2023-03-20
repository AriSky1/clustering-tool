# ClusteringApp


DASH / FLASK / POSTGRESQL / SCIKIT-LEARN

This app is a Data Science tool for Exploratory Data Analysis (EDA).<br><br>

https://user-images.githubusercontent.com/105871709/211107465-35405023-6ab1-4899-bc65-9822f5d8b841.mp4

Create, upload and manage datasets. Try random datasets from public domain.<br>
Apply the density based clustering algorithm to the data and download the output as a csv file.
<br>This version allows you to perform clustering on 3 numerical columns only.


The user can connect to personal PostgreSQL databases db and db2. <br>
The first one is supposed to be the place where user stores the created, uploaded and modified datasets. <br>
The second one is supposed to be a database which stores public cleaned datasets the user can analyze.<br>
<br>
The user can merge them into one by changing "db2" variable to "db" in main.py.<br>

Updates to come:
+ Allow deeper analysis with more features with dimensionality reduction techniques.<br>
+ Provide metrics
+ Provide an encoder for categorical values
