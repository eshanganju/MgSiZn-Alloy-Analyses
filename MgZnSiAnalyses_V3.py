"""
"""
from scipy.ndimage.morphology import distance_transform_edt as edt
import numpy as np
import tifffile as tf
from skimage.morphology import skeletonize
import skimage.measure
from scipy.spatial import ConvexHull
from scipy import ndimage
from skan import Skeleton, summarize
import pandas as pd
from numba import jit

VERBOSE = True

clm = '/home/eg/Desktop/EG-WateshedAnalysesAvizo/2023-01-28-D-3-8_LE_recon-borderKill.tif'	# input('Enter label map location: ')
ofl = '/home/eg/Desktop/EG-WateshedAnalysesAvizo/Zoom/' 									# input('Enter output folder location: ')
sampleName = 'Zoom'																			# input('Enter sample name: ')

genSTL=False
genEDM=False
genEDMSkeleton=False

genSkeleton=True
genPtclVol=True
genSaPtcl=True
genHullData = True


def convexHullDataOfParticle(particleMap, dilateParticle=False):
	"""Get convex hull of the particle
	"""
	particleMap = particleMap//particleMap.max()

	if dilateParticle == True:
		particleMap = ndimage.binary_dilation(particleMap,iterations = 1)

	particleData = np.transpose(np.where(particleMap==1))
	
	particleHull = ConvexHull(particleData)

	volume = particleHull.volume
	area = particleHull.area

	return area, volume

@jit(nopython=True)
def cropAndPadParticle(labelledMap,label,pad, saveData=True,fileName='',outputDir=''):
	"""
	"""
	loc = np.where(labelledMap == label)
	dimZ = int(loc[0].max() - loc[0].min())
	dimY = int(loc[1].max() - loc[1].min())
	dimX = int(loc[2].max() - loc[2].min())
	
	croppedParticle = labelledMap[ loc[0].min() : loc[0].max()+1,
									loc[1].min() : loc[1].max()+1,
									loc[2].min() : loc[2].max()+1 ]

	sizeZ = dimZ + 2 * pad
	sizeY = dimY + 2 * pad
	sizeX = dimX + 2 * pad

	paddedParticle = np.zeros( ( sizeZ,sizeY,sizeX ) )
	
	paddedParticle[ pad : dimZ+pad+1, pad : dimY+pad+1, pad : dimX+pad+1] = croppedParticle

	cleanedPaddedParticle = removeOtherLabels(paddedParticle, label)

	# convert label number to ones
	normalizedCleanedPaddedParticle = cleanedPaddedParticle//label

	return normalizedCleanedPaddedParticle

@jit(nopython=True)
def removeOtherLabels(paddedParticle, label):
	"""
	"""
	return np.where(np.logical_or(paddedParticle > label, paddedParticle < label), np.zeros_like(paddedParticle), paddedParticle)

def _generateInPlaceStlFile(labMap, stepSize = 1, saveImg=True, sampleName='', outputDir=''):
	"""Generate in place stl
	"""
	print('\tGenerating stl' )
	
	verts, faces, normals, values = marching_cubes( labMap, step_size=stepSize)

	mesh = trimesh.Trimesh(verts,faces)
	
	smoothMesh = trimesh.smoothing.filter_humphrey(mesh)
			
	sampleName = outputDir + sampleName + '.stl'
	smoothMesh.export(sampleName) 

def obtainEuclidDistanceMap( binaryMapForEDM, scaleUp = int(1), saveImg=False, sampleName='', outputDir='' ):
	"""Computes the euclidian distance tranform (EDT) for a binary map

	An EDT or an euclidian distance map (EDM) is an ndarray of the same size as
	the input binary map. Instead of 1 or 0, as in a binary map, each solid (1)
	voxel is assigned the distance between it and the nearest void (0) voxel.
	The resulting map shows how far each voxel is to the nearest void voxel,
	i.e. from the boundary of paricle.

	Parameters
	----------
	binaryMapForEDM : ndArray binary map of the xct scan

	scaleUp : unsigned integer
		This is done inorder to artificially increase the zoom of the data. The
		distance from the voxels increased if a fractional value is needed in
		the location of peaks.

	saveImg : bool
		If this is true, the edt is saved in the requested location or at the
		location whenere the code is run

	sampleName : string
		name of the sample - used to name the file used to store the edt data

	outputDir : string
		Location of the the ouput directory. If left empty, the file is saved at
		the same location as the location of the code.

	Returns
	-------
	edMap : ndarray
		array containing the euclidian distance map of the binary map

	"""
	if VERBOSE:
		print('\nFinding Euclidian distance map (EDM)')
		print('------------------------------------*')

	edMap = edt( binaryMapForEDM )

	if scaleUp!=0 :
		edMap =  edMap * scaleUp

	if VERBOSE: print( "EDM Created" )

	if saveImg == True:
		if VERBOSE: print('\nSaving EDM map...')
		tf.imsave(outputDir + sampleName + '-edm.tif',edMap)

	return edMap

def analyzeParticles(labMapLoc, sampleName='', saveData=True, outputDir='',):
	"""Code for the analyses of label map
	"""
	ofl=outputDir
	clm = tf.imread(labMapLoc).astype('uint16')
	numPtcl = clm.max()
	print('\nNum particles: ' + str(numPtcl))

	particleData = np.zeros((numPtcl,5)) 	
	#Index, Surface area, Hull area, volume, hull volume

	for ptclNo in range(1, numPtcl + 1):
		
		print('\nChecking particle ' + str(ptclNo) + '/' + str(numPtcl))	

		currFileName = sampleName + '-' + str(ptclNo)

		particleData[ptclNo-1,0] = ptclNo

		# Extract particle subvolume
		print('\tCropping')
		ptcl = cropAndPadParticle(labelledMap=clm,
											label=ptclNo,
											pad= 20,
											saveData=True,
											fileName= currFileName,
											outputDir=ofl)

		tf.imwrite( (ofl+currFileName+'.tiff'), ptcl.astype('uint8'))


		# Generate STL
		if genSTL == True:
			print('\tMaking Stl')
			_generateInPlaceStlFile( ptcl, 
										stepSize = 1, 
										saveImg=True, 
										sampleName=currFileName, 
										outputDir=ofl)

		# Getting particle volume
		if genPtclVol == True:
			print('\tGetting particle volume')
			ptclVol=np.sum(ptcl)
			particleData[ptclNo-1,3] = ptclVol
			print('\t\tVolume:', str(ptclVol))


		if genSkeleton == True and ptclVol >= 200:
			print('\tSkeletonizing')
			ptclSkeleton = skeletonize(ptcl)
			tf.imwrite( (ofl+currFileName+'-skeleton.tiff'), ptclSkeleton.astype('uint8'))

			# network analyses of skeleton
			dataForBranch = summarize(Skeleton(ptclSkeleton))
			dataForBranch.to_csv(ofl+currFileName+'-GraphData.csv',sep=',')

		else: print('\tParticle too small for skeleton...')


		# Compute EDM of particle subvolume
		if genEDM == True:
			print('\tEDMing')
			edmPtcl = obtainEuclidDistanceMap( binaryMapForEDM=ptcl, 
												scaleUp = int(1), 
												saveImg=False, 
												sampleName=currFileName, 
												outputDir=ofl )

		# Get product of skeleton EDM
		if genEDMSkeleton == True:
			print('\tGetting EDM on skeleton')
			skeletonEDM = ptclSkeleton*edmPtcl

			# Get list of ED along skeleton
			nonZeroEDMVal = (skeletonEDM[np.nonzero(skeletonEDM)]).flatten()
			np.savetxt(ofl+currFileName+'-edmSkeleton.csv', nonZeroEDMVal, delimiter=',')

		# Surface area of particles
		if genSaPtcl == True:
			print('\tGetting surface area of particle')
			vertices, faces, _, _ = skimage.measure.marching_cubes(volume=ptcl, 
																 		level=None, 
																 		spacing=(1.0, 1.0, 1.0),
																		gradient_direction='descent', 
																		step_size=1, 
																		allow_degenerate=True, 
																		method='lewiner', 
																		mask=None)
			
			particleData[ptclNo-1,1]= skimage.measure.mesh_surface_area(vertices, faces)
		
		# Hull Data
		if genHullData == True:
			print('\tGetting hull data')
			hullArea, hullVolume = convexHullDataOfParticle(ptcl, dilateParticle=True)
			particleData[ptclNo-1,2] = hullArea
			particleData[ptclNo-1,4] = hullVolume

		np.savetxt(ofl + sampleName+'-Data.csv',particleData,delimiter=',')

analyzeParticles(labMapLoc=clm, sampleName=sampleName, saveData=True, outputDir=ofl,)