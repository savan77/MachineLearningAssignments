import argparse
import pandas as pd


class KMeans:
	def __init__(self, K, data):
		self.K = K
		self.data = pd.read_csv(data, names=['tweet']) # tokenized text
		self.data = self.data['tweet'].sample(frac=1).reset_index(drop=True) #shuffle data in-place

	# follow sklearn API
	def fit(self, epochs):
		# initialize clusters randomly
		self.centroids = {}
		for i in range(self.K):
			self.centroids[i] = self.data[i]

		# infinite loop might not be a good idea so instead of stopping training after convergence
		# we will loop for some predefined epochs and if our model converes before `epochs` we will terminate training then
		for epoch in range(epochs):
			clusters = {i:[] for i in range(self.K)}
			for text in self.data:
				#calculate distance form each centroid
				distances = [self.jaccard(text, self.centroids[cent]) for cent in self.centroids]
				#find the centroid with min distance
				min_distance = distances.index(min(distances))
				clusters[min_distance].append(text)

			# find new cluster
			prev_centroids = self.centroids
			for cluster in clusters:
				self.centroids[cluster] = self.find_min(clusters[cluster])

			converged = True
			for cent in self.centroids:
				if self.jaccard(prev_centroids[cent], self.centroids[cent]) > 0.001:
					converged = False

			if converged:
				break



	def find_min(self):
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
		return sse_total



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--K",  default=5, help="number of clusters")
	parser.add_argument("--data", default="processed_data.csv")
	parser = parser.parse_args()
	km = KMeans(parser.K, parser.data)
	km.fit(3)
