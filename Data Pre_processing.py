"""
Data Pre-Processing for Machine learning Tasks
"""

# Import the required Modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split

class Data_Preprocessing():
	def __init__(self,data):
		self.dataset = pd.read_csv(data)
		self.X = self.dataset.iloc[:,:-1].values # Features
		self.y = self.dataset.iloc[:,-1].values  #Labels
		self.num_col = self.X.shape[1]
 

	# Taking care of missing data
	def imputer(self):
		#for i in range(self.num_col):
			#for j in self.X[:,i]:
		x = input("Input the strategy to fill the missing_values\n"
					"1.mean\n"
					"2.median\n"
					"3.most_frequent\n"
					"4.constant\n")
		if x == "constant":
			y = int(input("Enter an integer for the constant value :"))
			data_imputer = SimpleImputer(missing_values=np.nan,strategy=x,fill_value=y)
		elif x == "mean":
			data_imputer = SimpleImputer(missing_values=np.nan,strategy=x)
		elif x == "median":
			data_imputer = SimpleImputer(missing_values=np.nan,strategy=x)
		elif x == "most_frequent":
			data_imputer = SimpleImputer(missing_values=np.nan,strategy=x)
		self.X[:,1:3]=data_imputer.fit_transform(self.X[:,1:3])
		
		
	#OneHot encoding of categorical data
	def onehotencoding(self):
		Encoder = ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[0])],
				  remainder="passthrough")
		self.X = Encoder.fit_transform(self.X)
		
		
	# Encode the labels
	def labelencoder(self):
		label_encoder = LabelEncoder()
		self.y = label_encoder.fit_transform(self.y)
		

	# Split the data into train and test dataset
	def train_test_partition(self):
		while True:
			try:
				x = float(input("Input the test size :"))
				if x < 1:
					break
			except:
				continue
			print("Invalid test_size.It should be less than 1")
		self.X_train,self.X_test,self.y_train,self.y_test = \
			train_test_split(self.X,self.y,test_size=x,random_state=0)
		
	# Feature Scale the date by either Normalization or Standardisation
	def feature_scaling(self):
		sc = StandardScaler()
		self.X_train[:,3:5] = sc.fit_transform(self.X_train[:,3:5])
		self.X_test[:,3:5] = sc.fit_transform(self.X_test[:,3:5])
		print(self.X_train)

def main():
	# Create the Object Handle and run the required pre-processing technique
	data_preprocess = Data_Preprocessing("Data.csv")
	data_preprocess.imputer()
	data_preprocess.onehotencoding()
	data_preprocess.labelencoder()
	data_preprocess.train_test_partition()
	data_preprocess.feature_scaling()

if __name__ == "__main__":
	main()