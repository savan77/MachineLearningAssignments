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
			print("#############################################")
			print("Epoch: ", epoch)
			print("Centroid: ", self.centroids)
			self.clusters = {i:[] for i in range(self.K)}
			for text in self.data:
				
				#calculate distance form each centroid
				distances = [self.jaccard(text, self.centroids[cent]) for cent in self.centroids]
				#find the centroid with min distance
				min_distance = distances.index(min(distances))
				self.clusters[min_distance].append(text)

			# find new cluster

			prev_centroids = self.centroids.copy()
			for cluster in self.clusters:
				# print(cluster)
				self.centroids[cluster] = self.find_min(self.clusters[cluster])
			print("New", self.centroids)
			converged = True
			for cent in self.centroids:
				# print("cent", cent)
				# print(self.jaccard(prev_centroids[cent], self.centroids[cent]))
				if self.jaccard(prev_centroids[cent], self.centroids[cent]) > 0.001:
					# print("conv", converged)
					converged = False
					break
			print(self.sse())
			if converged:
				break

		print("----------------------------------------")
		print("K: ", self.K)
		print("SSE: ", self.sse())
		print([len(v) for c,v in self.clusters.items()])



	def find_min(self, clusters):
		distances = []
		dists_point = []
		for idx, point in enumerate(clusters):
			temp = 0
			for j in clusters[idx+1:]:
				temp += self.jaccard(point, j)
			distances.append(temp)
			dists_point.append(point)
		return dists_point[distances.index(min(distances))]

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
				sse_total += self.jaccard(self.centroids[i], self.clusters[i][j])
		return sse_total



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--K",  default=23, help="number of clusters")
	parser.add_argument("--data", default="processed_data.csv")
	parser = parser.parse_args()
	km = KMeans(parser.K, parser.data)
	km.fit(30)
