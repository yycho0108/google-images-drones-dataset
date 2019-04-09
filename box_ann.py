import os
import sys
import numpy as np
import cv2
from object_detection_tf import ObjectDetectorTF
from utils import draw_bbox

def main():
    app    = ObjectDetectorTF(use_gpu=True, model='model2')
    imgdir = '/tmp/ximg'
    viz    = False
    thresh = 0.95
    anndir = '/tmp/ximgann'
    batch  = 16
    shape  = (640,640)

    if not os.path.exists(anndir):
        os.makedirs(anndir)

    if viz:
        cv2.namedWindow('win', cv2.WINDOW_NORMAL)

    fs = os.listdir(imgdir)
    #np.random.shuffle(fs)
    #fs = ['/tmp/ximg/7703.jpg' for _ in range(batch)]
    fs = sorted(fs)

    idx = 0
    tot = len(fs)

    imgs = []
    sfs  = []
    for fi, f in enumerate(fs):
        f = os.path.join(imgdir, f)
        img = cv2.imread(f)
        if img is None:
            continue

        if batch > 1:
            # resize to input dimension for stacking
            img = cv2.resize(img, shape)

        imgs.append(img[...,::-1]) # remember: bgr->rgb
        sfs.append(f)
        if len(imgs) < batch:
            continue
        bimg = np.stack(imgs, axis=0) 
        bfs  = np.stack(sfs , axis=0)
        imgs = []
        sfs  = []

        res = app(bimg)

        cls, box, score = res['class'], res['box'], res['score']
        #print 'batch processing'
        for f1_,im1_,cls_,box_,score_ in zip(bfs, bimg, cls,box,score):
            msk = (score_ > thresh)
            if np.count_nonzero(msk) <= 0:
                continue

            cs  = cls_[msk]
            bs  = box_[msk]
            ss  = score_[msk]
            #print('scores', score)

            with open(os.path.join(anndir, '{}.txt'.format(idx)), 'w+') as fout:
                fout.write('{}\n'.format(f1_))
                fout.write('cx cy w h p\n')
                for b_, s_ in zip(bs, ss):
                    by0, bx0, by1, bx1 = b_
                    bcx = (bx0 + bx1) / 2.0
                    bcy = (by0 + by1) / 2.0
                    bw  = (bx1 - bx0)
                    bh  = (by1 - by0)
                    fout.write('{} {} {} {} {}\n'.format(bcx,bcy,bw,bh,s_))

            idx += 1
            if (idx % 100) == 0:
                print('{}:{}/{}'.format(idx,fi,tot))

            # visualization part
            if viz:
                for b_, c_ in zip(bs, cs):
                    #ry0,rx0,ry1,rx1 = box_ # relative
                    draw_bbox(im1_, b_, str(c_))
                print 'scores', ss
                cv2.imshow('win', im1_)
                k = cv2.waitKey(0)
                if k in [27, ord('q')]:
                    return

    if viz:
        cv2.destroyWindow('win')

if __name__ == "__main__":
    main()
