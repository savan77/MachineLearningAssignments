import argparse
import pandas as pd


class KMeans:
	def __init__(self, K, data):
		self.K = K
		self.data = pd.read_csv(data, names=['tweet'])
		self.data = self.data['tweet'] #tokenized tweets

	# follow sklearn API
	def fit(self, data):
		pass

	def jaccard(self, a, b):
		"""
			Jaccard_Dist = 1 - |(A M B)|/|(A U B)|
		"""
		common = list(set(a) & set(b))
		union = list(set(a) | set(b))
		return 1 - (len(common)/len(union))

	def sse(self):
		sse_total = 0
		for i in range(self.K):
			for j in range(len(self.clusters[i])):
				sse_total += self.jaccard(self.clusters[i][j]['centroid'], self.clusters[i][j])



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--K",  default=5, help="number of clusters")
	parser.add_argument("--data", default="processed_data.csv")
	parser = parser.parse_args()
	km = KMeans(parser.K, parser.data)
	# km.fit()