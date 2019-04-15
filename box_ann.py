#!/usr/bin/env python2
"""
Produce Bounding-Box annotations with a detector network.
"""
import os
import sys
import numpy as np
import cv2
from object_detection_tf import ObjectDetectorTF
from utils import draw_bbox
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
flags = tf.app.flags

flags.DEFINE_string('root', '/tmp',
        'Root Directory')
flags.DEFINE_string('model', 'model',
        'Model Directory')
flags.DEFINE_integer('batch_size', 16,
        'Number of Batches to run in parallel')
flags.DEFINE_boolean('viz', False,
        'Enable Runtime Visualization')
flags.DEFINE_float('thresh', 0.7,
        'Bounding-Box Detection Confidence Threshold')
flags.DEFINE_float('thresh2_lo', 0.5,
        'Secondary Bounding-Box Detection Confidence Threshold (disabled if <=thresh)')
flags.DEFINE_float('thresh2_hi', 0.85,
        'Secondary Bounding-Box Detection Confidence Threshold (disabled if <=thresh)')
flags.DEFINE_boolean('use_gpu', True,
        'Use GPU')
flags.DEFINE_string('img_dir', '/tmp/img',
        'Source Image Directory')
flags.DEFINE_string('out_dir', '/tmp/ann',
        'Output Annotations Directory')
FLAGS = flags.FLAGS

class TFRecordEnumerator(object):
    def __init__(self, path):
        self.path_  = path
        self.proto_ = {
                'image/height': tf.FixedLenFeature([], tf.int64),
                'image/width': tf.FixedLenFeature([], tf.int64),
                'image/filename': tf.FixedLenFeature([], tf.string),
                'image/source_id': tf.FixedLenFeature([], tf.string),
                'image/encoded': tf.FixedLenFeature([], tf.string),
                'image/format': tf.FixedLenFeature([], tf.string),
                'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
                'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
                'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
                'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
                'image/object/class/text': tf.VarLenFeature(tf.string),
                'image/object/class/label': tf.VarLenFeature(tf.int64)
                }

    def __call__(self):
        it  = tf.python_io.tf_record_iterator(path=rec_file)
        idx = 0
        for rec in it:
            ex  = tf.parse_single_example(rec, proto)
            f    = ex['image/filename'].numpy()
            xmin = tf.sparse.to_dense(ex['image/object/bbox/xmin']).numpy()
            xmax = tf.sparse.to_dense(ex['image/object/bbox/xmax']).numpy()
            ymin = tf.sparse.to_dense(ex['image/object/bbox/ymin']).numpy()
            ymax = tf.sparse.to_dense(ex['image/object/bbox/ymax']).numpy()
            yield idx, f, img
            idx += 1

class DirectoryEnumerator(object):
    def __init__(self, path, recursive=False):
        fs = os.listdir(path)
        fs = sorted(fs)
        n = len(fs)

        self.path_  = path
        self.fs_    = fs
        self.n_     = n

    def __call__(self):
        for idx, f in enumerate(self.fs_):
            f = os.path.join(self.path_, f)
            img = cv2.imread(f)
            if img is None:
                print 'None'
                continue
            yield idx, f, img

    def stat(self):
        print('FS len : {}'.format(self.n_))
        if self.n_ > 0:
            print('FS Sample : {}'.format(self.fs_[0]))

    def size(self):
        return self.n_

def expand_box(box, margin):
    y0, x0, y1, x1 = box
    cy = 0.5 * (box[...,0] + box[...,2])
    cx = 0.5 * (box[...,1] + box[...,3])

    center = np.stack([cy, cx, cy, cx], axis=-1)
    return (1.0 + margin)*(box - center) + center

def crop_box(img, box, margin=0.0):
    h, w = img.shape[:2]
    if margin > 0:
        box = expand_box(box, margin)
    box = np.clip(box, 0, 1)
    y0,x0,y1,x1 = np.multiply(box, [h,w,h,w]).astype(np.int32)
    return img[y0:y1,x0:x1], box

def sub_box(box, subbox):
    y0, x0, y1, x1 = box[...,0], box[...,1], box[...,2], box[...,3]
    bh = y1 - y0
    bw = x1 - x0
    o = np.stack([y0,x0,y0,x0], axis=-1) # origin (top-left)
    s = np.stack([bh,bw,bh,bw], axis=-1) # scale (hwhw)
    d = np.multiply(subbox, s) # delta (bottom-right)
    return o + d

def check_box(app, img, boxs, scores, thresh1, thresh2, shape, viz=True):
    good_msk = np.greater(scores, thresh2)
    bad_msk  = np.logical_and(thresh1<scores, ~good_msk)
    imgs = []
    boxs0 = []
    if np.any(bad_msk):
        for box in boxs[bad_msk]:
            sub_img, box = crop_box(img, box, margin=0.25)
            imgs.append(sub_img)
            boxs0.append( box )
    else:
        return good_msk, boxs, []
    boxs0 = np.float32(boxs0)

    if len(imgs) > 1:
        imgs  = [cv2.resize(si, shape)[...,::-1] for si in imgs]
    res   = app(imgs)

    idx   = np.argmax(res['score'], axis=-1)
    score = np.max(res['score'], axis=-1)

    if viz:
        for i, simg in enumerate(imgs):
            cv2.imshow('img-{}'.format(i), simg)
            cv2.waitKey(1)
        print '>>>>>>>', res['score']
        print 'score', score
        print 'salvage', (score > thresh2)

    bad_idx = np.argwhere(bad_msk)[:,0]
    #print 'bad_msk', bad_msk.shape
    #print 'bad_idx', bad_idx.shape
    #print bad_idx
    #print 'prv', good_msk
    good_msk[bad_idx] |= (score > thresh2)
    #print 'nxt', good_msk

    #print bad_idx[score > thresh2]

    boxs[bad_idx] = sub_box(boxs0, res['box'][np.arange(len(idx)), idx])
    return good_msk, boxs, bad_idx[score>thresh2]
    #bad_msk[bad_msk] &= (score < thresh2)
    #return ~bad_msk

def main(_):
    app    = ObjectDetectorTF(
            root=FLAGS.root,
            use_gpu=FLAGS.use_gpu,
            model=FLAGS.model
            )
    imgdir = FLAGS.img_dir
    viz    = FLAGS.viz
    thresh = FLAGS.thresh
    thresh2_lo = FLAGS.thresh2_lo
    thresh2_hi = FLAGS.thresh2_hi
    anndir = FLAGS.out_dir
    batch  = FLAGS.batch_size
    shape  = (640, 640) # TODO : figure this out from model directory pipeline.config??

    double_check = (
            thresh2_lo > 0 and
            thresh2_hi > 0 and
            thresh2_hi > thresh2_lo
            )

    if not os.path.exists(anndir):
        os.makedirs(anndir)
    if viz:
        cv2.namedWindow('win', cv2.WINDOW_NORMAL)

    oidx = 0

    imgs = []
    sfs  = []
    
    src = DirectoryEnumerator(imgdir)
    src.stat()
    tot = src.size()

    for (fidx, f, img) in src():
        if (fidx % 100) == 0:
            print('{}/{}: {}'.format(fidx, tot, oidx))

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
        for f1_,im1_,cls_,box_,score_ in zip(bfs, bimg, cls, box, score):
            salvaged_idx = []
            if (double_check):
                # double-checking thresh2 enabled
                msk, box_, salvaged_idx = check_box(
                        app, im1_, box_, score_,
                        thresh2_lo, thresh2_hi, shape, viz)
            else:
                msk = np.greater(score_, thresh)

            if np.count_nonzero(msk) <= 0:
                continue

            cs  = cls_[msk]
            bs  = box_[msk]
            ss  = score_[msk]

            with open(os.path.join(anndir, '{}.txt'.format(oidx)), 'w+') as fout:
                fout.write('{}\n'.format(f1_))
                fout.write('cx cy w h p\n')
                for b_, s_ in zip(bs, ss):
                    by0, bx0, by1, bx1 = b_
                    bcx = (bx0 + bx1) / 2.0
                    bcy = (by0 + by1) / 2.0
                    bw  = (bx1 - bx0)
                    bh  = (by1 - by0)
                    fout.write('{} {} {} {} {}\n'.format(bcx,bcy,bw,bh,s_))
            oidx += 1

            # visualization part
            if viz:
                print 'score', score_
                for b_, c_ in zip(bs, cs):
                    #ry0,rx0,ry1,rx1 = box_ # relative
                    draw_bbox(im1_, b_, str(c_))
                if len(salvaged_idx) > 0:
                    for b_, c_ in zip(box_[salvaged_idx], cls_[salvaged_idx]):
                        #ry0,rx0,ry1,rx1 = box_ # relative
                        draw_bbox(im1_, b_, str(c_), color=(0,0,255))
                #print 'scores', ss
                cv2.imshow('win', im1_)
                k = cv2.waitKey(0)
                if k in [27, ord('q')]:
                    return

    if viz:
        cv2.destroyWindow('win')

if __name__ == "__main__":
    tf.app.run()
