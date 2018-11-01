import argparse
import json
import os
import numpy as np

import utils

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator




def load_data(filename):
	return json.load(open(os.path.join('../data/', filename)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--do', required=True)

    io_args = parser.parse_args()
    action = io_args.do


    if action == 'load':
    	data = load_data('train.json')
    	utils.create_recipe_ingredients_matrix(data)

    if action == 'stats':
    	data = load_data('train.json')
    	X, y, recipe_mapper, ingredient_mapper, recipe_inverse_mapper, ingredient_inverse_mapper, cuisine_mapper, cuisine_inverse_mapper = utils.load(data)
    	print("All loaded")
    	
    	count = np.bincount(y)
    	# for i in range(20):
    	# 	print(cuisine_inverse_mapper[i],":",count[i])
    	fig = plt.figure()
    	plt.bar(list(range(20)), count, align='edge', tick_label=list(cuisine_mapper.keys()))
    	plt.xticks(rotation=50)
    	plt.savefig('../figs/cuisine-count.jpg')
    	plt.close()

    	count = count//200
    	text = ''
    	for i in range(20):
    		text += count[i]*(cuisine_inverse_mapper[i]+' ')
    	
    	wordcloud = WordCloud(max_words=20, background_color="white", collocations=False).generate(text)
    	plt.figure()
    	plt.imshow(wordcloud, interpolation="bilinear")
    	plt.axis("off")
    	plt.savefig('../figs/cuisine-cloud.jpg')
    	
    	
