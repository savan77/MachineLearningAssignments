import pandas as pd
import argparse

def preprocess_util(text):
	""" processes each tweet """
	# 2 - remove any word that starts from @
	# 3 - rmeove any hastab symbol '#'
	# 4 - remove URL
	# 5 - Convert every word to lowercase
	text_2 = filter(lambda x: x[0] != '@', text.split())
	text_4 = filter(lambda x: x[:7] != 'http://', text_2)
	text_3 = [w[1:].lower().strip("'") if w[0] == "#" else w.lower().strip("'") for w in text_4]
	return text_3


def preprocess(file):
	data = pd.read_csv(file, sep="|", names=['id', 'timestamp', 'tweet'])
	# 1 - remove id and timestamp
	data = data.drop(['id','timestamp'], axis=1)
	data = data['tweet'].apply(preprocess_util)
	# print(data.head())
	return data
	

def save_to_file(data, out_file):
	data.to_csv(out_file)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--file",  default="Health-Tweets/bbchealth.txt",help="path to txt file to be processed")
	parser.add_argument("--sep",  default="|",help="seperator to be used while reading txt file")
	parser.add_argument("--out_file", default="processed_data.csv")
	parser = parser.parse_args()
	data = preprocess(parser.file)
	save_to_file(data, parser.out_file)