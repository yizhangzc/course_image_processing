import argparse
import os
import time

import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import (extract_patches_2d,
                                              reconstruct_from_patches_2d)

import guidedfilter


def path_check( path ):

    if not os.path.isfile( path ):
        print( "can't find the image, please check the image path!" )
        exit(0)

def rain_snow_detection( img ):

    img_size    = img.shape[0: 2]

    idx_x   = np.reshape( 
        np.tile( np.reshape( np.arange( img_size[0] ), [-1, 1] ), 
            [ 1, img_size[1] ] ), (-1) )

    idx_y   = np.reshape( 
        np.tile( np.reshape( np.arange( img_size[1] ), [1, -1] ), 
            ( img_size[0], 1 )), (-1) )

    lower_bound = np.zeros( 4, dtype = np.int32 )
    upper_bound = np.array( [ img_size[0], img_size[0], img_size[1], img_size[1] ], dtype = np.int32 )
    idx     = np.array( [[  [ i-6,  i+1,    j-6,    j+1 ],
                            [ i-6,  i+1,    j,      j+7 ],
                            [ i-3,  i+4,    j-3,    j+4 ],
                            [ i,    i+7,    j-6,    j+1 ],
                            [ i,    i+7,    j,      j+7 ] ] for i, j in zip( idx_x, idx_y ) ] )
    idx = np.where( idx >= lower_bound, idx, lower_bound )
    idx = np.where( idx <= upper_bound, idx, upper_bound )

    # import pdb; pdb.set_trace()
    
    five_means = np.array( 
        [ [ np.mean( img[ xl:xu, yl:yu, : ], axis = (0,1) ) 
        for xl, xu, yl, yu in win ] for win in idx ] )

    loc_map = np.sum( np.greater( 
        np.reshape( img, ( -1, 1, 3 ) ), five_means ).astype( np.int32 ), axis = ( 1, 2 ) )
    loc_map = np.reshape( np.less( loc_map, 15 ), img_size )

    # io.imshow( loc_map )
    # plt.show( )

    return loc_map
    
def img_decomposition( img, loc_map ):
    
    img_size    = img.shape[0: 2]

    non_dynamic_img = np.multiply( img, np.expand_dims( loc_map, 2 ) )
    filled_non_dynamic_img = non_dynamic_img.copy()
    
    zero_idx = np.nonzero( loc_map == 0 )

    lower_bound = np.zeros( 4, dtype = np.int32 )
    upper_bound = np.array( [ img_size[0], img_size[0], img_size[1], img_size[1] ], dtype = np.int32 )

    # import pdb; pdb.set_trace()

    for i, j in zip( zero_idx[0], zero_idx[1] ):
        idx = np.array( [ i-4, i+5, j-4, j+5 ] )
        idx = np.where( idx >= lower_bound, idx, lower_bound )
        idx = np.where( idx <= upper_bound, idx, upper_bound )
        xl, xu, yl, yu = idx

        filled_non_dynamic_img[i,j] = np.true_divide( 
            np.sum( non_dynamic_img[ xl:xu, yl:yu, : ], axis = (0, 1) ),
            np.sum( loc_map[ xl:xu, yl:yu ], axis = ( 0, 1 ) ) )
    
    # import pdb; pdb.set_trace()
    # img_show( filled_non_dynamic_img )

    low_freq_part   = np.concatenate( [ np.expand_dims( guidedfilter.guidedfilter(
        filled_non_dynamic_img[:,:,c], filled_non_dynamic_img[:,:,c], 4, 0.0001 ),
            2 ) for c in range(3) ], axis = 2 ).astype( int )

    # img_show( low_freq_part )
    # guide_filter

    return low_freq_part

def dictionary_learning( high_freq_part ):
    
    patch_size  = ( 16, 16 )
    data        = extract_patches_2d( high_freq_part ,patch_size )
    # http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.extract_patches_2d.html
    data        = data.reshape( data.shape[0], -1 ).astype( float )
    data        -= np.mean( data, axis = 0 )
    data        /= np.std( data, axis = 0 )

    dic0        = MiniBatchDictionaryLearning( n_components = 1024, alpha = 1, 
                                                n_iter = 100, fit_algorithm = 'lars' )
    V           = dic0.fit( data ).components_

    import pdb; pdb.set_trace()


def img_show( img ):
    io.imshow( img )
    plt.show()

def main( path ):

    path_check( path )
    img = io.imread( path )

    loc_map = rain_snow_detection( img )

    low_freq_part   = img_decomposition( img, loc_map )
    high_freq_part  = np.subtract( img, low_freq_part )

    dictionary_learning( high_freq_part )

    # img_show( low_freq_part )



if __name__ == '__main__':
    
    parser  = argparse.ArgumentParser(description="snow/rain removing")
    parser.add_argument('-p', '--path', type=str, default = "bird.jpg", help='the image path', required = True)
    args    = parser.parse_args()

    main( args.path )
