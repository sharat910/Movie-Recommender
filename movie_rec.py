import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

def fetch_data():
	return fetch_movielens(min_rating=4.0)

def sample_recommendation(model, data, user_ids):

	#Number of users and movies in the training data
	n_users, n_items = data['train'].shape

	# Generate recommendations for each user we input
	for user_id in user_ids:

		#Movies they already like
		known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

		#Movies our model predicts
		scores = model.predict(user_id,np.arange(n_items))
		
		#rank them in order of most liked to least
		top_items = data['item_labels'][np.argsort(-scores)]

		#Print the results
		print "User %s" % user_id

		print "Movies liked by user: "

		for x in known_positives[:3]:
			print "\t"+ str(x)

		print "Recommendations:"

		for x in top_items[:3]:
			print "\t" + str(x)


if __name__ == '__main__':
	
	#data fetch 
	data = fetch_data()

	#Test prints
	print repr(data['train'])
	print repr(data['test'])

	# Create model
	model = LightFM(loss='warp') #Weighted Approximate-Rank Pairwise Algo | Hybrid (Content + Collaborative)

	# Train Model
	model.fit(data['train'],epochs=30, num_threads=4)

	# Recommender
	sample_recommendation(model,data,[1,2,3])


