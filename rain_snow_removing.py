__author__ = 'yizhangzc'

# course:     CV
# teacher:    lxq
# author:     zju_cs / Yi Zhang / 21721190
# mail:       yizhangzc@gmail.com
# date:       2018/5
# environment:  ubuntu 14.04 / python 3.5 / numpy 1.14 /

import argparse
import math
import os
import time

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy import misc
from scipy.ndimage.filters import median_filter
from skimage import feature as ft
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import (extract_patches_2d,
                                              reconstruct_from_patches_2d)

import utils


class RSRM( object ):

    def __init__( self, img_path ):
        self._img_path      = img_path
        self._patch_size    = (8, 8 )


    def path_check( self ):
        if not os.path.isfile( self._img_path ):
            print( "can't find the image, please check the image path!" )
            exit(0)

    def rs_detection( self, img ):
        img_size    = img.shape[ 0: 2 ]

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

        five_means = np.array( 
            [ [ np.mean( img[ xl:xu, yl:yu, : ], axis = (0,1) ) 
            for xl, xu, yl, yu in win ] for win in idx ] )

        loc_map = np.sum( np.greater( 
            np.reshape( img, ( -1, 1, 3 ) ), five_means ).astype( np.int32 ), axis = ( 1, 2 ) )
        loc_map = np.reshape( np.greater( loc_map, 14 ), img_size )

        return loc_map


    def load_img( self ):
        
        self.path_check()
        img = Image.open( self._img_path )
        
        self._origin_img        = np.asarray( img )
        self._origin_img_size   = self._origin_img.shape

    def img_decomposition( self ):
        
        dynamic_loc = self.rs_detection( self._origin_img )
        non_dynamic_loc = np.logical_not( dynamic_loc ).astype( int )

        non_dynamic = np.multiply( self._origin_img, np.expand_dims( non_dynamic_loc, 2 ) )
        filled_non_dynamic  = non_dynamic.copy()

        img_size    = self._origin_img_size[0: 2]
        zero_idx    = np.nonzero( non_dynamic_loc == 0 )
        lower_bound = np.zeros( 4, dtype = np.int32 )
        upper_bound = np.array( [ img_size[0], img_size[0], img_size[1], img_size[1] ], dtype = np.int32 )

        # import pdb; pdb.set_trace()

        for i, j in zip( zero_idx[0], zero_idx[1] ):
            idx = np.array( [ i-4, i+5, j-4, j+5 ] )
            idx = np.where( idx >= lower_bound, idx, lower_bound )
            idx = np.where( idx <= upper_bound, idx, upper_bound )
            xl, xu, yl, yu = idx

            non_dynamic_sum = np.sum( non_dynamic[ xl:xu, yl:yu, : ], axis = (0, 1) )
            non_dynamic_num = np.sum( non_dynamic_loc[ xl:xu, yl:yu ], axis = ( 0, 1 ) )
            if non_dynamic_num <= 0:
                non_dynamic_num = 1
            filled_non_dynamic[i,j] = np.divide( non_dynamic_sum, non_dynamic_num )

        # import pdb; pdb.set_trace()

        self._low_freq_part     = np.concatenate( [ np.expand_dims( utils.guidedfilter(
            filled_non_dynamic[:,:,c], filled_non_dynamic[:,:,c], 4, 0.0001 ),
                2 ) for c in range(3) ], axis = 2 ).astype( int )

        self._high_freq_part    = np.subtract( self._origin_img, self._low_freq_part )

    def dictionary_learning( self ):

        patch_size  = self._patch_size
        
        data        = extract_patches_2d( self._high_freq_part, patch_size )
        data        = data.reshape( data.shape[0], -1 )
        random_idx  = np.arange( data.shape[0] )
        np.random.shuffle( random_idx)
        
        training_data   = data[ random_idx[0: 20000] ]

        # import pdb; pdb.set_trace()
        dic = MiniBatchDictionaryLearning( n_components = 1024, alpha = 0.15,
                                    n_iter = 10, fit_algorithm = 'lars', n_jobs = 8, transform_algorithm = 'lars' )

        dic.fit( training_data )
        # dic.set_params(  )
        # coefs   = dic.transform( data )

        return dic

        # import pdb; pdb.set_trace()
    

    def img_cut( self, img ):

        patch_size  = self._patch_size

        multi0  = math.ceil( img.shape[0] / patch_size[0] )
        multi1  = math.ceil( img.shape[1] / patch_size[1] )

        new_img     = np.zeros( (  multi0 * patch_size[0], multi1 * patch_size[1], 3 ) )
        new_img[ :img.shape[0] , :img.shape[1], : ] = img

        idx_x   = np.reshape( 
            np.tile( np.reshape( np.arange( multi0 ) * patch_size[0], [-1, 1] ), 
                [ 1, multi1 ] ), (-1) )
        idx_y   = np.reshape( 
            np.tile( np.reshape( np.arange( multi1 ) * patch_size[1], [1, -1] ), 
                ( multi0, 1 )), (-1) )

        data = [ np.reshape( new_img[ i:i+patch_size[0], j:j+patch_size[1], : ], (-1) ) 
            for i, j in zip( idx_x, idx_y ) ]

        return np.array( data )

    def img_splice( self, patches, size ):
        
        # import pdb; pdb.set_trace()

        patch_size  = self._patch_size
        
        multi0  = math.ceil( size[0] / patch_size[0] )
        multi1  = math.ceil( size[1] / patch_size[1] )

        new_img = np.zeros( ( multi0 * patch_size[1], multi1 * patch_size[1], 3 ) )

        idx_x   = np.reshape( 
            np.tile( np.reshape( np.arange( multi0 ) * patch_size[0], [-1, 1] ), 
                [ 1, multi1 ] ), (-1) )
        idx_y   = np.reshape( 
            np.tile( np.reshape( np.arange( multi1 ) * patch_size[1], [1, -1] ), 
                ( multi0, 1 )), (-1) )
        idx_p   = np.arange( np.size( idx_x ) )

        for i, j, p in zip( idx_x, idx_y, idx_p ):
            new_img[ i:i+patch_size[0], j:j+patch_size[1], : ] = patches[p]

        return new_img[ :size[0], :size[1], : ]


    def detail_extract_l1( self ):
        
        patch_size  = self._patch_size
        threshold1  = 0.0035
        threshold2  = 0.048
        threshold3  = 2

        D       = self.dictionary_learning()
        atoms   = D.components_

        data    = self.img_cut( self._high_freq_part )
        coefs   = D.transform( data )

        # import pdb; pdb.set_trace()

        atoms   = atoms.reshape( (atoms.shape[0], patch_size[0], patch_size[1], 3) )

        # import pdb; pdb.set_trace()
        sum_variance    = np.mean( np.var( atoms, 3 ), axis = (1, 2) )
        hor_gradient    = np.mean( np.absolute( np.gradient( atoms,  axis = 1 ) ), axis = (1, 2, 3) )

        non_dynamic     = sum_variance > threshold1
        non_dynamic     = np.logical_or( non_dynamic, hor_gradient < threshold2 )
        dynamic         = np.logical_not( non_dynamic )


        HoG             = np.array( [ ft.hog( atom[:,:,1], orientations = 9, 
                                    pixels_per_cell = (8,8), cells_per_block = (1, 1) ) for atom in atoms ])

        pdip            = np.array( [ np.where( h == max(h) )[0][0] for h in HoG ] )

        non_dynamic     = np.logical_or( non_dynamic, np.absolute( pdip - np.mean( pdip[dynamic] ) ) > threshold3 )

        # import pdb; pdb.set_trace()

        modified_coefs  = np.multiply( coefs, non_dynamic )
        patches         = np.dot( modified_coefs, D.components_ )
        patches         = patches.reshape( -1, patch_size[0], patch_size[1], 3 )

        recover         = self.img_splice( patches, self._origin_img_size )

        self._img_detail_l1 = np.where( recover >= 0., recover, 0. )
        self._high_freq_l1  = self._high_freq_part - self._img_detail_l1


    def detail_extract_l2( self ):

        # import pdb; pdb.set_trace()
        
        dynamic_loc = self.rs_detection( self._high_freq_part )
        non_dynamic_loc = np.logical_not( dynamic_loc ).astype( int )

        non_dynamic = np.multiply( self._high_freq_l1, np.expand_dims( non_dynamic_loc, 2 ) )
        filled_non_dynamic  = non_dynamic.copy()

        img_size    = self._origin_img_size[0: 2]
        zero_idx    = np.nonzero( non_dynamic_loc == 0 )
        lower_bound = np.zeros( 4, dtype = np.int32 )
        upper_bound = np.array( [ img_size[0], img_size[0], img_size[1], img_size[1] ], dtype = np.int32 )        

        for i, j in zip( zero_idx[0], zero_idx[1] ):
            idx = np.array( [ i-4, i+5, j-4, j+5 ] )
            idx = np.where( idx >= lower_bound, idx, lower_bound )
            idx = np.where( idx <= upper_bound, idx, upper_bound )
            xl, xu, yl, yu = idx

            non_dynamic_sum = np.sum( non_dynamic[ xl:xu, yl:yu, : ], axis = (0, 1) )
            non_dynamic_num = np.sum( non_dynamic_loc[ xl:xu, yl:yu ], axis = ( 0, 1 ) )
            if non_dynamic_num <= 0:
                non_dynamic_num = 1
            filled_non_dynamic[i,j] = np.divide( non_dynamic_sum, non_dynamic_num )

        detail = np.concatenate( [ np.expand_dims( utils.guidedfilter(
            filled_non_dynamic[:,:,c], filled_non_dynamic[:,:,c], 3, 0.0001 ), 2 ) 
                for c in range(3) ], axis = 2 ).astype( int )

        self._img_detail_l2 = np.where( detail >= 0., detail, 0. )

        self._high_freq_l2  = np.subtract( self._high_freq_l1, self._img_detail_l2 )

    def detail_extract_l3( self ):

        img_size    = self._high_freq_part.shape[ 0: 2 ]
        img         = self._high_freq_part

        idx_x   = np.reshape( 
            np.tile( np.reshape( np.arange( img_size[0] ), [-1, 1] ), 
                [ 1, img_size[1] ] ), (-1) )
        idx_y   = np.reshape( 
            np.tile( np.reshape( np.arange( img_size[1] ), [1, -1] ), 
                ( img_size[0], 1 )), (-1) )
        
        lower_bound = np.zeros( 4, dtype = np.int32 )
        upper_bound = np.array( [ img_size[0], img_size[0], img_size[1], img_size[1] ], dtype = np.int32 )
        idx     = np.array( [ [ i-5,  i+6,    j-5,    j+6 ] for i, j in zip( idx_x, idx_y ) ] )
        idx = np.where( idx >= lower_bound, idx, lower_bound )
        idx = np.where( idx <= upper_bound, idx, upper_bound )        

        A_matrix = np.array( [ np.mean( img[ xl:xu, yl:yu, : ], axis = (0,1) ) for xl, xu, yl, yu in idx ] )
        A_matrix = A_matrix.reshape( ( self._origin_img_size ) )

        # import pdb; pdb.set_trace()

        A_matrix = median_filter( A_matrix, size = ( 7, 7, 1 ) )

        V_  = np.var( A_matrix, 2 )
        V   = np.power( V_ / np.max( V_ ), 1.1 )

        detail = np.multiply( np.expand_dims( V, 2 ), self._high_freq_part )

        self._img_detail_l3 = np.where( detail >= 0, detail, 0. )

    def show_result( self ):
        self._result = self._low_freq_part + self._img_detail_l1.astype(int) + self._img_detail_l2.astype(int) + self._img_detail_l3.astype(int)
        
        misc.imsave( 'detail_l1.png', ( self._img_detail_l1 ).astype(int) )
        misc.imsave( 'detail_l2.png', ( self._img_detail_l2 ).astype(int) )
        misc.imsave( 'detail_l3.png', ( self._img_detail_l3 ).astype(int) )
        misc.imsave( 'low_freq.png',  (self._low_freq_part).astype(int) )
        misc.imsave( 'high_freq.png', (self._high_freq_part).astype(int) )

        misc.imsave( 'low_detail1.png',  self._low_freq_part + self._img_detail_l1.astype(int) )
        misc.imsave( 'low_detail12.png', self._low_freq_part + self._img_detail_l1.astype(int) + self._img_detail_l3.astype(int) )
        misc.imsave( 'result.png',       self._result )

        # import pdb; pdb.set_trace()


if __name__ == '__main__':
    
    parser  = argparse.ArgumentParser(description="snow/rain removing")
    parser.add_argument('-p', '--path', type=str, default = "bird.jpg",
                            help='the image path', required = True)

    args    = parser.parse_args()
    rsrm = RSRM( args.path )
    
    rsrm.load_img()
    rsrm.img_decomposition()
    rsrm.detail_extract_l1()
    rsrm.detail_extract_l2()
    rsrm.detail_extract_l3()
    rsrm.show_result()