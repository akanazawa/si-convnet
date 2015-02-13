import numpy as np
from scipy import ndimage
from scipy import misc
from scipy import signal
from matplotlib.pyplot import *
import math

def getImageCoords(img):
    return genCoordMat(img.shape[0], img.shape[1])

def genCoordMat(rows, cols):
    coords = np.zeros((rows * cols, 3))
    for index in xrange(coords.shape[0]):
        coords[index, 0] = index / cols #y
        coords[index, 1] = index % cols #x
        coords[index, 2] = 1

    return coords

def applyTransMat(coords, trans):
    return coords * trans

def transMat(dx, dy):
    return np.asarray([[1, 0, 0], [0, 1, 0], [dx, dy, 1]])

def rotMat(angle):
    return np.asarray([[np.cos(angle), np.sin(angle), 0], [-np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

def scaleMat(sx, sy):
    return np.asarray([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])

def homogenDivMat(coords):
    return coords.ElementDivideBroadcast(m[:, 2])



def interpImageNN_clip(img, coordMat):
    rows = img.shape[0]
    cols = img.shape[1]
    result = np.ones((rows, cols))
    assert rows * cols == coordMat.shape[0]
    for index in xrange(coordMat.shape[0]):
        r = index / cols
        c = index % cols
        (sampR, sampC,) = coordMat[index, :2]
        result[r, c] = img[sampR, sampC]

    return result


def interpImageNN_full(img, rows, cols, coordMat):
    result = np.zeros((rows, cols, 3)) if len(img.shape)==3 \
             else np.zeros((rows, cols))
    old_row, old_col = img.shape
    assert rows * cols == coordMat.shape[0]
    usingSub = 1 if coordMat.shape[1] > 1 else 0
    for index in xrange(coordMat.shape[0]):
        r = index / cols
        c = index % cols
        if usingSub:
            (sampR, sampC,) = coordMat[index, :2]
            sampR = round(sampR)
            sampC = round(sampC)
        else:
            sampR = int(coordMat[index]) / old_col
            sampC = int(coordMat[index]) % old_col
            # (wantR, wantC) = coordMat_sub[index, :2]
            # wantR = int(wantR)
            # wantC = int(wantC)
            # if int(wantR) != sampR or int(wantC) != sampC:
            #     import ipdb as pdb; pdb.set_trace()
        if (sampR < old_row and sampR >= 0) \
              and (sampC < old_col and sampC >= 0):                        
          result[r, c] = img[sampR, sampC]
          
    return result

def interpImageBilin(img, coordMat):
    rows = img.shape[0]
    cols = img.shape[1]
    result = np.ones((rows, cols))
    assert rows * cols == coordMat.shape[0]
    for index in xrange(coordMat.shape[0]):
        r = index / cols
        c = index % cols
        (sampR, sampC,) = coordMat[index, :2]
        fracX = sampR - np.floor(sampR)
        fracY = sampC - np.floor(sampC)
        sampR = np.floor(sampR)
        sampC = np.floor(sampC)
        sampR2 = min(sampR + 1, rows - 1)
        sampC2 = min(sampC + 1, cols - 1)
        yVal1 = (1 - fracY) * img[sampR, sampC] + fracY * img[sampR, sampC2]
        yVal2 = (1 - fracY) * img[sampR2, sampC] + fracY * img[sampR2, sampC2]
        val = (1 - fracX) * yVal1 + fracX * yVal2
        result[r, c] = val

    return result

def interpImage(img, coordMat):
    return interpImageBilin(img, coordMat)

def addTransform(m, transform):
    return np.dot(m, transform)

def clampCoords(coords, rows, cols):
    for index in xrange(coords.shape[0]):
        coords[index, 0] = max(0, min(coords[index, 0], rows - 1))
        coords[index, 1] = max(0, min(coords[index, 1], cols - 1))

    return coords

def reflectCoords(coords, rows, cols):
    def reflect(val, N):
        if val < 0:
            val = -np.floor(val)
            val = val % (2 * N - 2)
        if val >= N:
            if val > 2*N -2: print "larger!!!"
            val = 2 * N - 2 - val
        return val

    for index in xrange(coords.shape[0]):
        r = int(reflect(coords[index, 0], rows))
        c = int(reflect(coords[index, 1], cols))
        # r = int(reflect(coords[index, 0]+0.5, rows))
        # c = int(reflect(coords[index, 1]+0.5, cols))

        coords[index, 0] = r
        coords[index, 1] = c

    return coords

def coordSub2Ind(coordMat, width):
    coordInd = np.zeros((coordMat.shape[0], 1));
    for i in xrange(coordMat.shape[0]):
        (r, c) = coordMat[i, :2]
        if (r < 0 or c < 0 ):
            coordInd[i] = -1
        else :
            index = int(r)*width + int(c)
            coordInd[i] = index

    return coordInd

def get_new_size(rows, cols, m):
    #4 corners in row col 1 order
    # corners = np.asarray([[0, 0, 1], [0, cols, 1], [rows, 0, 1], [rows, cols, 1] ])
    # # left multiply
    # new_corners = np.dot(corners, m)
    # # new_row = int(math.ceil(np.max(new_corners[:,0])) - np.floor(np.min(new_corners[:,0])))
    # # new_col = int(math.ceil(np.max(new_corners[:,1])) - np.floor(np.min(new_corners[:,1])))
    # new_row = int(np.max(new_corners[:,0]) - np.min(new_corners[:,0])+0.5)
    # new_col = int(np.max(new_corners[:,1]) - np.min(new_corners[:,1])+0.5)
    # offset_y = np.min(new_corners[:,0])    
    # offset_x = np.min(new_corners[:,1])
    # in x, y, z
    corners = np.asarray([[0, 0, 1], [cols, 0, 1], [0, rows, 1], [cols, rows, 1] ])
    # right multiply
    new_corners = np.dot(m, corners.T).T
    new_row = int((max(new_corners[:,1])) - (min(new_corners[:,1])))
    new_col = int((max(new_corners[:,0])) - (min(new_corners[:,0])))
    offset_y = np.min(new_corners[:,1])    
    offset_x = np.min(new_corners[:,0])
    
    return (new_row, new_col, offset_x, offset_y)

def getSize(rows, cols, angle):
    angle = -angle * math.pi / 180
    matrix = [
    math.cos(angle), math.sin(angle), 0.0,
    -math.sin(angle), math.cos(angle), 0.0
    ]
    def transform(x, y, matrix=matrix):
        (a, b, c, d, e, f) = matrix
        return a*x + b*y + c, d*x + e*y + f

    # calculate output size
    w = cols
    h = rows
    xx = []
    yy = []
    for x, y in ((0, 0), (w, 0), (w, h), (0, h)):
        x, y = transform(x, y)
        xx.append(x)
        yy.append(y)
    w = int(math.ceil(max(xx)) - math.floor(min(xx)))
    h = int(math.ceil(max(yy)) - math.floor(min(yy)))
    print "new w %d new h %d" %(w,h)


def getCoordMatAll_noinv(nrow, ncol, scale, rad):
    m_inv = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    m_fwd = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # These order must flip
    m_inv = addTransform(m_inv, rotMat(-rad))
    m_inv = addTransform(m_inv, scaleMat(1./scale, 1./scale))

    m_fwd = addTransform(m_fwd, scaleMat(scale, scale))
    m_fwd = addTransform(m_fwd, rotMat(rad))

    # obtain size of new image
    new_row, new_col, _, _ = get_new_size(nrow, ncol, m_fwd)

    # gen coord of the new image
    coords = genCoordMat(new_row, new_col)
    ## offset:
    new_center = np.asarray([(new_row-1)/2.0, (new_col-1)/2.0]) 
    old_center = np.asarray([(nrow-1)/2.0, (ncol-1)/2.0 ]) 
    # remove offset:
    coords[:,:1]  = coords[:,:1] - new_center[0]
    coords[:,1:2]  = coords[:,1:2] - new_center[1]

    newCoords = np.dot(coords, m_inv)
    # back in
    # add the translation:
    newCoords[:,:1]  = newCoords[:,:1] + old_center[0] 
    newCoords[:,1:2]  = newCoords[:,1:2] + old_center[1] 
    
    newCoords = newCoords + 0.0000001
    return (newCoords, new_row, new_col, m_inv)

# same  as _noinv but how C++ is implemented:
def getCoordMatAll_noinv2(nrow, ncol, scale, rad):
    m_fwd = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    m_fwd = addTransform(m_fwd, scaleMat(scale, scale))
    m_fwd = addTransform(m_fwd, rotMat(rad))
    m_inv = np.linalg.inv(m_fwd)
    # obtain size of new image
    new_row, new_col, _, _ = get_new_size(nrow, ncol, m_fwd)

    # gen coord of the new image
    coords = genCoordMat(new_row, new_col)
    ## offset:
    new_center = np.asarray([(new_row-1)/2.0, (new_col-1)/2.0])
    # new_center = np.asarray([(new_row)/2.0, (new_col)/2.0]) 
    #old_center = np.asarray([(nrow)/2.0, (ncol)/2.0 ])
    old_center = np.asarray([(nrow-1)/2.0, (ncol-1)/2.0 ]) 
    # remove offset:
    m_inv = addTransform(transMat(-new_center[0], -new_center[1]), m_inv)

    newCoords = np.dot(coords, m_inv)
    # back in
    # add the translation:
    newCoords[:,:1]  = newCoords[:,:1] + old_center[0]
    newCoords[:,1:2]  = newCoords[:,1:2] + old_center[1]

    return (newCoords, new_row, new_col, m_inv)

def demo():
    img = misc.imread('/nfshomes/kanazawa/Pictures/cat.jpg', flatten=1)
    img = misc.imresize(img, 0.5)
    #    img = img[:, 0:-1]
    print "original img size (%d %d)" % (img.shape[0], img.shape[1])
    fignum = 4;
    figure(fignum); clf;
    subplot(221)
    imshow(img, cmap=cm.gray)
    title('original')
    gridWidth = 2
    gridHeight = 2
    numImages = gridWidth * gridHeight - 1
    
    scale = np.asarray([1, 0.85, 2, 4]);# 1.0 + (np.random.rand(numImages, 2) - 0.5) * 1.5
    rot = np.asarray([90, -15, 0.0, -90.0]) * (np.pi / 180.0)
    flip = np.zeros(numImages)#np.random.randint(2, size=numImages)
    for i in xrange(numImages):
        newCoords, new_row, new_col, _ = getCoordMatAll_noinv2(img.shape[0], img.shape[1], scale[i], rot[i])
        # reflectCoords(newCoords, img.shape[0], img.shape[1])
        interpImg = interpImageNN_full(img, new_row, new_col, newCoords)
        numpy_img = img;
        numpy_img = ndimage.interpolation.rotate(numpy_img, rot[i]*180./np.pi, order=0);
        numpy_img = misc.imresize(numpy_img, scale[i], interp='nearest');
        print "mine (%d, %d) numpy (%d, %d)" % (interpImg.shape[0], \
                                                interpImg.shape[1], \
                                                numpy_img.shape[0],\
                                                numpy_img.shape[1])
        # assert(interpImg.shape[1]== numpy_img.shape[1])
        figure(fignum)
        subplot(2,2,2+i)
        imshow(interpImg, cmap=cm.gray)
        params = "scale %.3g, rot %.3g deg flip %d" %\
          (scale[i],rot[i]*(180/np.pi)*100,flip[i])
        title(params)
        figure(fignum+1)
        subplot(2,2,2+i)
        imshow(numpy_img, cmap=cm.gray)
        if interpImg.shape == numpy_img.shape:
            print "diff %.4g" % np.sqrt(np.sum((interpImg - numpy_img)**2))
            figure(fignum+2)
            subplot(2,2,2+i)
            imshow(interpImg - numpy_img, cmap=cm.gray)
            title("diff %.4g" % np.sqrt(np.sum((interpImg - numpy_img)**2)));

def test(h, w, scale, angle, reflect = 0, no_inv = 1, mode='check'):
    # img = np.zeros((height, width))
    # w = 4
    # h = 3
    angle = angle* (np.pi/180)
    scale = float(scale);
    # scale = 1.5
    if mode is 'check':    
        img = checkerboard(w, h)
    else:
        # img = misc.lena()
        # img = misc.imresize(img, 0.3)
        img = np.random.rand(h, w)
        
    figure(2)
    subplot(131)
    imshow(img, cmap=cm.gray)
    title('original')
    
    if no_inv:
        newCoords, new_row, new_col, tmat = getCoordMatAll_noinv2(img.shape[0], img.shape[1], scale, angle)
    else:
        newCoords, new_row, new_col, tmat = getCoordMatAll(img.shape[0], img.shape[1], scale, angle)

    newCoords = np.round(newCoords);    
    if reflect:
        reflectCoords(newCoords, img.shape[0], img.shape[1])
    else:
        #clampCoords(newCoords, img.shape[0], img.shape[1])
        for index in xrange(newCoords.shape[0]):
            row = newCoords[index, 0]
            col = newCoords[index, 1]
            if ( row >= img.shape[0] or row < 0 ) or (col >= img.shape[1] or (col < 0 ) ) :
                newCoords[index, 0] = -1
                newCoords[index, 1] = -1

    newCoords_ind = coordSub2Ind(newCoords, img.shape[1])
    for i in xrange(len(newCoords_ind)):
        print "%d, " % newCoords_ind[i],
        if ((i+1) % new_col) == 0:
            print ""

    print "h_old = %d; w_old = %d " % (img.shape[0], img.shape[1])
    print "h_new = %d; w_new = %d " % (new_row, new_col)
    interpImg = interpImageNN_full(img, new_row, new_col, newCoords_ind)
    numpy_img = img;
    # numpy_img = misc.imrotate(numpy_img, angle*180/np.pi, interp='nearest');
    numpy_img = ndimage.interpolation.rotate(numpy_img, angle*180/np.pi, order=0);
    numpy_img = misc.imresize(numpy_img, scale, interp='nearest');
    # matrix = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # matrix = addTransform(matrix, scaleMat(scale, scale))
    # matrix = addTransform(matrix, rotMat(angle))
    # numpy_img = ndimage.interpolation.affine_transform(img, \
    #         matrix[0:2, 0:2], output_shape = (new_row, new_col), order = 1)
    
    # plot
    subplot(132)
    imshow(interpImg, cmap=cm.gray)
    params = "scale %.3g, rot %.3g deg" % (scale,angle*(180/np.pi)*100)
    title(params)
    subplot(133)
    imshow(numpy_img, cmap=cm.gray)
    title('with numpy')
    print "mine (%d, %d) numpy (%d, %d)" % (interpImg.shape[0], \
                                            interpImg.shape[1], \
                                            numpy_img.shape[0],\
                                            numpy_img.shape[1])
    return (newCoords, img, interpImg, numpy_img, tmat)

def checkerboard(w, h):
    re = np.r_[ w*[0, 1] ]
    ro = re^1
    return np.row_stack(h*(re,ro))

def interpolate_image(img, scale, angle):
    angle = angle * (np.pi/180.)
    newCoords, new_row, new_col, tmat = getCoordMatAll_noinv2(img.shape[0], img.shape[1], scale, angle)
    # reflectCoords(newCoords, img.shape[0], img.shape[1])
    newCoords = coordSub2Ind(newCoords, img.shape[1])
    
    interpImg = interpImageNN_full(img, new_row, new_col, newCoords)

    return interpImg

def crop_center(I, height, width):
    if I.shape[0] < height:
        I_new = np.zeros((height, width))
        # offset = np.floor( (height - I.shape[0] - 1 ) / 2 )
        offset = np.floor( (height - I.shape[0] ) / 2 )
        test = I_new[offset:offset + I.shape[0], offset: offset + I.shape[1] ]
        I_new[offset:offset + I.shape[0], offset: offset + I.shape[1] ] = I
    else:
        center = np.floor( (np.asarray(I.shape) - 1) /2. )
        offset = np.floor( (height - 1) / 2. )
        I_new = I[center[0]-offset:center[0] + offset + (height - 1) % 2 + 1,
                  center[1]-offset:center[1] + offset + (height - 1) % 2 + 1]
        # I_new = I[center[0]-offset:center[1] + offset + 1 + ( I.shape[0] - 1 ) % 2, 
        #           center[1]-offset:center[1] + offset + 1 + ( I.shape[0] - 1 ) % 2]
    assert(I_new.shape == (height, width))    
    return I_new

# that does crop_center at newCoords level
def interpolate_image_crop_center(img, scale, angle, height, width):
    angle = angle * (np.pi/180.)
    newCoords, new_row, new_col, tmat = getCoordMatAll_noinv2(img.shape[0], img.shape[1], scale, angle)
    # reflectCoords(newCoords, img.shape[0], img.shape[1])
    newCoords_ind = coordSub2Ind(newCoords, img.shape[1])
    coords_cropped = crop_center_coordinates(newCoords_ind, new_row, new_col, height, width)
    interpImg = interpImageNN_full(img, height, width, coords_cropped)
    return interpImg

def crop_center_coordinates(coords, height, width, target_h, target_w):
    if height < target_h:
        coords_new = -np.ones((target_h*target_w, 1))
        offset = np.floor( (target_h - height ) / 2 )
        start_r = offset
        end_r = offset + height
        start_c = offset
        end_c = offset + width
        counter = 0
        for row in xrange(int(start_r), int(end_r)):
            for col in xrange(int(start_c), int(end_c)):
                ind = row*target_w + col                
                coords_new[ind] = coords[counter]
                # coords_new[ind, :] = coords[counter, :2]
                counter += 1
        # for r in xrange(10):
        #     for c in xrange(10):
        #         ind = r*target_w + c
        #         print "%d, " % coords_new[ind],
        #     print ""        
    else:
        coords_new = np.zeros((target_h*target_w, 1))                    
        center = np.floor( (np.asarray([height, width]) - 1) /2. )
        offset = np.floor( (target_h - 1) / 2. )
        start_r = center[0] - offset;
        end_r = center[0] + offset + (target_h - 1) % 2 + 1;
        start_c = center[1] - offset;
        end_c = center[1] + offset + (target_h - 1) % 2 + 1;

        counter = 0;
        for row in xrange(int(start_r), int(end_r)):
            for col in xrange(int(start_c), int(end_c)):
                ind = row*width + col
                # coords_new[counter, :] = coords[ind, :2]
                coords_new[counter] = coords[ind]
                counter += 1
                
    print "start %d end %d" % (start_r, end_r)
    return coords_new
        
def downpool_test(mode = 'lena'):
    if mode is 'lena':
        # img = checkerboard(10, 10)
        # img = misc.imresize(img, 5.0, interp='nearest');
        img = misc.lena()
        img = misc.imresize(img, 0.8)
    else:
        img = np.float64(misc.imread('/nfshomes/kanazawa/Pictures/cat.jpg', flatten=1))
        img = misc.imresize(img, 0.5)
        img = img[0:img.shape[0], 0:img.shape[0]]

    # (scale, angle(degrees)) pair
    # trans = [ (2.,0), (0.5, 0.), (1., 15.), (2., -45.),(0.75, -25) ]
    trans = [ (0.25,0), (0.5, 0.), (0.5, 15.), (0.75, -10.)]
    
    # convolution filter:
    # filt = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
    filt = np.ones((3,3))/9.
    cano = signal.convolve2d(img, filt, mode='valid')

    height, width = cano.shape
    
    figure(20)
    subplot(221)
    imshow(img, cmap=cm.gray)
    title('original')
    subplot(222)
    imshow(np.maximum(cano, 0), cmap=cm.gray)
    title('canonical response')
    subplot(223)
    imshow(filt, cmap=cm.gray)
    title('filter')

    results = np.zeros( (height, width, len(trans)) )
    
    figure(30)
    counter = 0
    for scale, angle in trans:
        I_here = interpolate_image(img, scale, angle)
        subplot(3, len(trans), counter+1)
        imshow(I_here, cmap=cm.gray)
        params = "scale %.3g, rot %.3g" % (scale, angle)
        title(params)

        I_here = signal.convolve2d(I_here, filt, mode='valid')
        subplot(3, len(trans), len(trans) + counter+1)
        imshow(np.maximum(I_here, 0), cmap=cm.gray)

        ## Backward
        I_back0 = interpolate_image(I_here, 1./scale, -angle)
        I_back0 = crop_center(I_back0, height, width)
        # I_back = I_back0
        I_back = interpolate_image_crop_center(I_here, 1./scale, -angle, height, width)
        assert( np.all(I_back0 == I_back) )
        
        subplot(3, len(trans), 2*len(trans) + counter+1)
        imshow(np.maximum(I_back, 0), cmap=cm.gray)
        draw()
        assert(I_back.shape == cano.shape)

        results[:,:,counter] = I_back
        
        counter += 1

    figure(20)
    subplot(224)
    imshow(np.maximum(np.amax(results, axis=2), 0), cmap=cm.gray)
    title('max response')

    return (results, cano)

def test_crop_center_coordinates(size, cano_size, scale=1., angle=0.):
    # (newCoords, _, new_img, _, _) = test(size, size, \
    #                                  scale, angle, reflect=1, mode='rand')

    # newCoords_ind = coordSub2Ind(newCoords, size)

    coords = genCoordMat(size, size)    
    coords_ind = coordSub2Ind(coords, size)
    
    want = crop_center_coordinates(coords_ind, size, size, cano_size, cano_size)

    # cropped = interpolate_image_crop_center(new_img, 1/scale, -angle, cano_size, cano_size)

    # figure(5)
    # imshow(cropped, cmap = cm.gray);
    for i in xrange(len(want)):
        print "%d, " % want[i],
        if ((i+1) % cano_size) == 0:
            print ""
            
    # return (want, coords)

