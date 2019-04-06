import numpy as np
import os
import cv2

def main():
    droot = os.path.expanduser('~/libs/drone-net/label')
    img_root = '/media/ssd/datasets/drones/drone-net-img'

    idx = 0
    for f in os.listdir(droot):
        imf = '.'.join(f.split('.')[:-1]) + '.jpg'
        with open('/media/ssd/datasets/drones/drone-net-ann/{:05d}.txt'.format(idx), 'w+') as of:
            of.write(os.path.join(img_root, imf) + '\n')
            of.write('cx cy w h p\n')
            #print os.path.join(img_root, imf)
            h, w = cv2.imread(os.path.join(img_root, imf)).shape[:2]
            boxs = open(os.path.join(droot, f)).readlines() # cx cy w h, abs
            boxs = [[float(e) for e in s.split(' ')[1:]] for s in boxs]
            boxs = np.divide(boxs, [[w,h,w,h]])
            for b in boxs:
                bcx, bcy, bw, bh = b
                of.write('{} {} {} {} {}\n'.format(bcx,bcy,bw,bh,1))
        idx += 1

if __name__ == '__main__':
    main()
