#!/usr/bin/env python
"""
Using custom colors
===================
Using the recolor method and custom coloring functions.
"""

import numpy as np
import pandas as pd
from PIL import Image
from os import path
import matplotlib.pyplot as plt
import random

from wordcloud import WordCloud, STOPWORDS


def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)

d = path.dirname(__file__)

# read the mask image
# taken from
# http://www.stencilry.org/stencils/movies/star%20wars/storm-trooper.gif
mask = np.array(Image.open(path.join(d, "eikona.jpg")))

# movie script of "a new hope"
# http://www.imsdb.com/scripts/Star-Wars-A-New-Hope.html
# May the lawyers deem this fair use.
#text = open(path.join(d, './datasets/train_set.csv')).read()

text1 = ""
text2 = ""
text3 = ""
text4 = ""
text5 = ""
data = pd.read_csv('./datasets/train_set.csv', sep="\t")
for i in range(data.shape[0]):
	if(data['Category'][i] == "Football"):
		text1 +=  data['Content'][i]
	elif(data['Category'][i] == "Politics"):
		text2 +=  data['Content'][i]
	elif(data['Category'][i] == "Film"):
		text3 +=  data['Content'][i]
	elif(data['Category'][i] == "Business"):
		text4 +=  data['Content'][i]
	else:
		text5 +=  data['Content'][i]

for i in range(5):
	if(i == 0):
		text = text1;
		to_file_name = "Football-wordcloud.png"
	elif(i == 1):
		text = text2
		to_file_name = "Politics-wordcloud.png"
	elif(i == 2):
		text = text3
		to_file_name = "Film-wordcloud.png"
	elif(i == 3):
		text = text4
		to_file_name = "Business-wordcloud.png"
	else:
		text = text5
		to_file_name = "Technology-wordcloud.png"
	# preprocessing the text a little bit
	text = text.replace("HAN", "Han")
	text = text.replace("LUKE'S", "Luke")

	# adding movie script specific stopwords
	stopwords = set(STOPWORDS)

	wc = WordCloud(max_words=1000, mask=mask, stopwords=stopwords, margin=10,
               random_state=1).generate(text)
	# store default colored image
	default_colors = wc.to_array()
	plt.title("Custom colors")
	#plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3),
    #       interpolation="bilinear")
	wc.to_file(to_file_name)
	plt.axis("off")
	plt.figure()
	plt.title("Default colors")
	plt.imshow(default_colors, interpolation="bilinear")
	plt.axis("off")
	plt.show()
