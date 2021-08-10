# Original license for TestTargetFinder:
# The Leginon software is Copyright under
# Apache License, Version 2.0
# For terms of the license agreement
# see http://leginon.org
#
# New modifications made by Mariusz Matyszewski
# Also licensed under Apache License, Version 2.0
# see read.me in the github repo for more information

import os.path
import threading
from leginon import leginondata
from pyami import mrc
import targetfinder
import gui.wx.TestTargetFinder
#new imports
import numpy as np
import tensorflow as tf
from scipy.signal import convolve

def find_best(inp):
    #finds the best combination of exposures and focus points using a Monte Carlo search
    #designed with the Arctica Talos in mind, with a fixed beam size
    #modifications required to more effectively use on a Titan Krios
    res = inp[0] #matrix containing classifier output
    indices = inp[1] #index of carbon areas
    inv_scores = inp[2] #how close to the center the carbon is. Using 1/R to calculate value (R of 0 is impossible due to shape of the matrix, but keep that in mind if changing search grid size)
    iter_size = inp[3] #number of iterations to try
    seed = inp[4] #fixed seed. Leftover from mutlithreading variant
    #original version used multithreading, hence a packed input rather than passing them individually
    local_rng = np.random.RandomState(seed)
    best = 0 #variable to store best value
    best_points = [] #variable to store the best set of points
    l = [[-2,0],[0,2],[2,0],[0,-2]] #determines how big the beam burn should be. Uses a star shape. Can be removed for the Titan
    for n in range(iter_size):
        points = []
        total = 0
        cur_ver = res.copy() #makes a copy of the matrix to make changes too
        padded = np.pad(cur_ver, 2, mode='constant')
        car_area = indices[local_rng.choice(len(indices), p=inv_scores)] #randomly picks a carbon focus point, weighted by how close it is to the center
        padded[car_area[0]+1:car_area[0]+4,car_area[1]+1:car_area[1]+4] = -100 #simulated beam burn to the general area
        for j in l: #burns the sourrounding area
            padded[car_area[0]+j[0]+2,car_area[1]+j[1]+2] = -100
        cur_ver = padded[2:16,2:16] #updates cur_ver to contain the burn
        points.append(car_area) #adds focus point to the list. Focus point will always be first
        for i in range(16): #tries to add up to 16 points, but will usually stop before then.
            conv = convolve(cur_ver, np.ones((3,3)), mode='same', method='direct') #convolves the burn to make locations near it less attractive to the algorithm
            maxes = np.argsort(conv,axis=None,)[::-1][0:5] #finds the locations of the max values
            maxes = [np.unravel_index(i,(14,14)) for i in maxes]
            max_sums = np.array([conv[i] for i in maxes])
            t_val = max_sums > 0 #only keeps the values if value is above 0 (not burned), sets to false otherwise and will be skipped in the next step
            maxes = [i[0] for i in zip(maxes,t_val) if i[1]]
            max_sums = max_sums[t_val]
            if len(max_sums) == 0:
                break
            max_sums = max_sums/sum(max_sums) #normalized to 0 to work for Monte Carlo search
            ind = maxes[local_rng.choice(len(max_sums),p=max_sums)] #chooses a point weighted by its goodness score
            if conv[ind] > 0:
                # tmp_sum = total + conv[ind]
                # if conv[ind]/tmp_sum < 0.1:
                #     break
                # points.append(ind)
                # total = tmp_sum
                # above is an alternative code that used the conv values. Using non conv works better
                tmp_sum = total + cur_ver[ind]
                if cur_ver[ind]/tmp_sum < 0.1:
                    break
                points.append(ind)
                total = tmp_sum
                padded[ind[0]+1:ind[0]+4,ind[1]+1:ind[1]+4] = -100 #simulates new burn
                for j in l:
                    padded[ind[0]+j[0]+2,ind[1]+j[1]+2] = -100
                cur_ver = padded[2:16,2:16]
            else:
                break
        if total > best: #updates best score if better
            best = total
            best_points = points
    return (best, best_points)


class TestTargetFinder(targetfinder.TargetFinder):
    #most code left unchanged from the Leginon version. Only "your_targetfinder" is changed significantly.
	panelclass = gui.wx.TestTargetFinder.Panel
	settingsclass = leginondata.TestTargetFinderSettingsData
	defaultsettings = dict(targetfinder.TargetFinder.defaultsettings)
	defaultsettings.update({
		'test image': '',
	})
	def __init__(self, *args, **kwargs):
		self.userpause = threading.Event()
		targetfinder.TargetFinder.__init__(self, *args, **kwargs)
		self.image = None
		self.model = tf.keras.models.load_model("full_model_all_filters_v3") #imports tensorflow model
		self.start()

	def readImage(self, filename=''):
		if filename:
			self.image = mrc.read(filename)
		self.setImage(self.image, 'Image')

	def testFindTargets(self):
		focus_targets_on_image = []
		acquisition_targets_on_image = []

		# Put your function call here:
		# such as
		focus_targets_on_image, acquisition_targets_on_image = self.your_targetfinder(self.image)

		# here is an example of the output
		# focus_targets_on_image = [(50,20)]
		# acquisition_targets_on_image = [(100,100),(200,100)]

		self.setTargets(acquisition_targets_on_image, 'acquisition')
		self.setTargets(focus_targets_on_image, 'focus')
		import time
		time.sleep(1)

		if self.settings['user check']:
			self.panel.foundTargets()

	def your_targetfinder(self, image):
		image_org = np.array(image)
		image_norm = (image_org - np.mean(image_org))/np.std(image_org)
		image_norm = (image_norm + 7.5)/15
		image_norm[image_norm < 0] = 0
		image_norm[image_norm > 1] = 1
        #above code normalizes the image the same way as during training

		ten_arr = np.zeros((14*14,64,64), dtype='float32')
		for i in range(14):
			for j in range(14):
				n = i*14 + j
				ten_arr[n] = image_norm[64*j+14:64*(j+1)+14,64*i+14:64*(i+1)+14]
		ten_arr = ten_arr.reshape(-1, 64, 64, 1)
        #image is split up into segments

		pred = self.model.predict(ten_arr) #uses model to predict classification scores

		pred = pred.reshape(14,14,4)

		tmp = np.moveaxis(pred, 2, 0) #changes order of axis to make it easier to convolve
		c_fil = np.array([[[2,2,2],[2,5,2],[2,2,2]],[[-1,-1,-1],[-1,-2,-1],[-1,-1,-1]],[[-2,-2,-2],[-2,-5,-2],[-2,-2,-2]],[[0,0,0],[0,-2,0],[0,0,0]]])
        #c_fil is the convolution instructions. It gives different weights to different classes. Fil is good, agg, ice, and carbon is bad, but with varying degrees.
		res_1 = convolve(tmp[0], c_fil[0], mode='same')
		res_2 = convolve(tmp[1], c_fil[1], mode='same')
		res_3 = convolve(tmp[2], c_fil[2], mode='same')
		res_4 = convolve(tmp[3], c_fil[3], mode='same')
		res = (res_1+res_2+res_3+res_4)
        #convolution done seperately for each class, then added together
		car_np = pred[:,:,3]
		car_dist = np.linalg.norm(np.argwhere(car_np>0.8)-[6.5,6.5],axis=1)
        #finds predicted carbon areas with scores of 0.8+ then centers them to 6.5,6.5. Decimals used to make sure it never returns 0. Doesn't have to be absolute center
		inv_scores = [1/i for i in car_dist] #inverses the scores to be used for Monte Carlo search
		if len(inv_scores) == 0:
			inv_scores = [1]
			indices = [(7,7)]
            #if there is no carbon, then just return center as carbon point.
		else:
			inv_scores = inv_scores/sum(inv_scores)
			indices = [(i[0],i[1]) for i in np.argwhere(car_np>0.8)]

		cpu = 1 #does not work. Multithreading has a memory leak and had to be disabled
		total_iter = 5000 #How many Monte Carlo simulation to do
		split = total_iter // cpu

		in_vec = (res, indices, inv_scores, split, np.random.randint(100000)) #leftover from multithreading

		# with Pool(cpu) as p:
		# 	results = p.starmap(find_best, in_vec)
        # causes memory leak; disabled

		sorted_res = find_best(in_vec) #runs the Monte Carlo search

		centers = [(i[0]*64+46, i[1]*64+46) for i in sorted_res[1][1:]] #converts the point for leginon. Skips location 0 as that is the focus point
		if len(centers) == 0:
			focus = []
            #if no points, then don't submit a focus target
		else:
			focus = [(sorted_res[1][0][0]*64+46,sorted_res[1][0][1]*64+46)]
            #focus target is the first point in the returned list

		return focus, centers

	def findTargets(self, imdata, targetlist):
		image = imdata['image']

		self.setImage(image, 'Image')

		self.image = image
		self.testFindTargets()

		if self.settings['user check']:
			# user now clicks on targets
			self.notifyUserSubmit()
			self.userpause.clear()
			self.setStatus('user input')
			self.userpause.wait()

		self.setStatus('processing')

		self.publishTargets(imdata, 'focus', targetlist)
		self.publishTargets(imdata, 'acquisition', targetlist)

		self.logger.info('Targets have been submitted')

	def targetTestImage(self):
		usercheck = self.settings['user check']
		self.settings['user check'] = False
		filename = self.settings['test image']
		try:
			image = mrc.read(filename)
		except:
			self.logger.error('Failed to load test image')
			raise
			return
		self.setImage(image, 'Image')

		self.image = image
		self.testFindTargets()

		self.settings['user check'] = usercheck
