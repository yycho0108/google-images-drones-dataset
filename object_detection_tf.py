#!/usr/bin/env python2

import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import cv2
import time

from utils import draw_bbox

def expand_box(box, margin):
    y0, x0, y1, x1 = [box[..., i] for i in range(4)]
    cy = 0.5 * (y0 + y1)
    cx = 0.5 * (x0 + x1)
    center = np.stack([cy, cx, cy, cx], axis=-1)
    return (1.0 + margin)*(box - center) + center

def crop_box(img, box, margin=0.0):
    h, w = img.shape[:2]
    if margin > 0:
        box = expand_box(box, margin)
    box = np.clip(box, 0.0, 1.0)
    y0,x0,y1,x1 = np.multiply(box, [h,w,h,w]).astype(np.int32)
    return img[y0:y1,x0:x1], box

def inner_box(box, subbox):
    y0, x0, y1, x1 = [box[..., i] for i in range(4)]
    bh = y1 - y0
    bw = x1 - x0
    o = np.stack([y0,x0,y0,x0], axis=-1) # origin (top-left)
    s = np.stack([bh,bw,bh,bw], axis=-1) # scale (hwhw)
    d = np.multiply(subbox, s) # delta (bottom-right)
    return o + d

class ObjectDetectorTF(object):
    """
    Thin wrapper around the Tensorflow Object Detection API.
    Heavily inspired by the [provided notebook][1].

    Note:
        [1]: https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
    """

    # human-friendly label map
    mmap_ = {
            'coco' : 'ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03',
            'model2-drone-300x300' : '1voCDNghyKCNzk7Q9c4fzj23lW5jQtoTp'
            }

    def __init__(self,
            root='/tmp',
            model='model',
            gpu=0.0,
            threshold=0.5,
            threshold2=None,
            cmap=None,
            max_batch=8
            ):
        """
        Arguments:
            root(str)               : Persistent data directory; override to avoid initialization overhead.
            model(str)              : Model name; refer to the [model zoo][2].
            gpu(float)              : Enables execution on the GPU; specifies fraction of gpu to occupy.
            threshold(float)        : Minimum bounding-box confidence score to be considered valid.
            threshold2(A(2, float)) : Parameters weak-strong two-step validation; `None` to disable.
            cmap(dict)              : Alternative class definitions; remap known classes for convenience.
            max_batch(int)          : Maximum number of batches for stacked execution. Important for CPU execution.
        Note:
            [2]: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
        """

        # cache arguments
        self.root_  = root
        self.model_ = model
        self.gpu_ = float(gpu) # implicitly converted to float
        self.threshold_ = threshold
        self.threshold2_ = threshold2
        self.max_batch_ = max_batch

        # load model
        ckpt_file = self._maybe_download_ckpt()
        self.graph_ = self._load_graph(ckpt_file)
        self.input_, self.output_, self.shape_ = self._build_pipeline(self.graph_)
        self.cmap_ = cmap


        # print configuration
        print("""
        -- [Object Detection Model Config] --
        root       : {}
        model      : {}
        ckpt       : {}
        gpu        : {}
        threshold  : {}
        threshold2 : {}
        max batch  : {}
        shape      : {}
        --------------------------------------
        """.format(root, model, ckpt_file, gpu, threshold, threshold2, max_batch,
            self.shape_))

        self.initialize()

    def __call__(self, img):
        """
        Run object detection.

        Arguments:
            img(A(N?,H,W,3, uint8)): image to run inference on; may optionally be batched.
        Returns:
            output(dict): {'box':A(N?,M,4), 'class':A(N?,M), 'score':A(N?,M)}
        """
        if np.ndim(img) == 3:
            # single image configuration
            outputs = self.__call__(img[None,...])
            return {k:v[0] for k,v in outputs.iteritems()}
        
        if len(img) > self.max_batch_:
            # to keep reasonable memory footprint,
            # split to multiple parts; kind of important when gpu:=false
            n_split = int(np.ceil(len(img)/float(self.max_batch_)))
            imgs    = np.array_split(img, n_split, axis=0)
            outputs = {}
            for imgs_i in imgs:
                out_i = self.__call__(imgs_i)
                for (k, v) in out_i.items():
                    if k in outputs:
                        outputs[k] = np.concatenate([outputs[k], v])
                    else:
                        outputs[k] = v
            return outputs

        # default configuration
        outputs = self.run(self.output_, {self.input_:img})
        if self.cmap_ is not None:
            # map to alternative class definitions
            outputs['class'] = [[ str(self.cmap_[x] if (x in self.cmap_) else x) for x in xs] for xs in outputs['class']]
            outputs['class'] = np.array(outputs['class'], dtype=str)
        return outputs

    def detect(self, img,
            is_bgr=True,
            threshold=None,
            threshold2=None
            ):
        """
        Wrapper around ObjectDetectorTF.__call__() that handles convenience arguments,
        such as two-step detection and channel conversion.
        TODO: currently only works for non-batch style inputs.

        Arguments:
            img(A(H,W,3, uint8))    : image to run inference on; batch-mode disabled.
            is_bgr(bool)            : whether `img` is organized channelwise rgb or bgr.
            threshold(float)        : minimum bounding-box confidence score to be considered valid.
            threshold2(A(2, float)) : parameters weak-strong two-step validation; `None` to disable.
        returns:
            output(dict)            : {'box':A(M, 4), 'class':A(M), 'score':A(M)}
        """

        # supply default args
        if threshold is None:
            threshold = self.threshold_
        if threshold2 is None:
            threshold2 = self.threshold2_

        if np.ndim(img) > 3:
            print('batch configuration not currently supported for ObjectDetectorTF.detect()')
            return None

        if is_bgr:
            # reverse final axis, bgr -> rgb
            img = img[..., ::-1]
        # if is_normalized:
        #    # convert from 0-1 range to 0-255
        #    img = (img * 255).astype(np.uint8)
        res = self.__call__(img)
        cls, box, score = res['class'], res['box'], res['score']

        if threshold2 is None or (threshold2[0] >= threshold):
            # double-checking disabled or pointless
            msk = (threshold <= score)
            #return cls[msk], box[msk], score[msk]
            return {k:v[msk] for (k,v) in zip(
                ['class', 'box', 'score'],
                [cls, box, score])}
        else:
            # double-checking enabled
            good_msk = (threshold <= score)
            retry_msk = np.logical_and.reduce([
                    threshold2[0] <= score,
                    #score < threshold2[1],
                    ~good_msk
                    ])

            if not np.any(retry_msk):
                # no box to retry - early return
                return {k:v[good_msk] for (k,v) in zip(
                    ['class', 'box', 'score'],
                    [cls, box, score])}

            sub_imgs = []
            sup_clss = []
            sup_boxs = []
            sup_scores = []
            for c, b, s in zip(cls[retry_msk], box[retry_msk], score[retry_msk]):
                sub_img, sup_box = crop_box(img, b, margin=0.25)
                sub_imgs.append(cv2.resize(sub_img, self.shape_[:2][::-1]))
                sup_boxs.append( sup_box )
                sup_clss.append( c )
                sup_scores.append( s )
            sub_imgs = np.stack(sub_imgs, axis=0)
            sub_res = self.__call__(sub_imgs) # TODO : consider `recursive` detection

            new_cls = []
            new_box = []
            new_score = []

            for (spc, spb, sps, sbcs, sbbs, sbss) in zip(
                    sup_clss, sup_boxs, sup_scores,
                    sub_res['class'], sub_res['box'], sub_res['score']
                    ):
                for (sbc, sbb, sbs) in zip(sbcs, sbbs, sbss):
                    if sbs < threshold2[1]:
                        # stricter score threshold
                        continue
                    if sbc != spc:
                        # matching class
                        continue
                    new_cls.append( spc )
                    new_box.append( inner_box(spb, sbb) )
                    #new_score.append( ( sps, sbs ) )
                    new_score.append(sps)
            new_box = np.reshape(new_box, (-1,4))

            # finalize result
            fin_cls = np.concatenate([cls[good_msk], new_cls], axis=0)
            fin_box = np.concatenate([box[good_msk], new_box], axis=0)
            fin_score = list( score[good_msk] )
            fin_score.extend( new_score )
            #np.concatenate([score[good_msk], new_score], axis=0)

            return {k:v for (k,v) in zip(
                ['class', 'box', 'score'],
                [fin_cls, fin_box, fin_score])}
        # should never reach here
        return None

    def _download_from_gdrive(self):
        try:
            from google_drive_downloader import GoogleDriveDownloader as gd
            gd.download_file_from_google_drive(
                    file_id=ObjectDetectorTF.mmap_[self.model_],
                    dest_path=os.path.join(self.root_, self.model_, 'frozen_inference_graph.pb'),
                    unzip=True)
        except Exception as e:
            print('Downloading From GDrive Failed : {}'.format(e))

    def _download_from_tfzoo(self,
            model_tar_basename,
            model_tar_fullpath
            ):
        # fetch from web if tar file does not exist
        if not os.path.exists(model_tar_fullpath):
            print('downloading model ...'.format(self.model_))
            download_base = 'http://download.tensorflow.org/models/object_detection/'
            opener = urllib.request.URLopener()
            opener.retrieve(download_base+model_tar_basename, model_tar_fullpath)

        # extract if graph does not exist
        tar_file = tarfile.open(model_tar_fullpath)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, self.root_)

    def _maybe_download_ckpt(self):
        """
        WARN: internal.

        Check if model file exists; download if not available.
        Returns:
            ckpt_file(str): path to the checkpoint, from which to load graph.
            TODO : actually .pb file
        """
        ckpt_file = os.path.join(self.root_, self.model_,
                'frozen_inference_graph.pb')
        if not os.path.exists(ckpt_file):
            if self.model_ in ObjectDetectorTF.mmap_:
                # fetch from google drive
                self._download_from_gdrive()
            else:
                # fetch from tf zoo
                model_tar_basename = '{}.tar.gz'.format(self.model_)
                model_tar_fullpath = os.path.join(self.root_, model_tar_basename)
                self._download_from_tfzoo(
                        model_tar_basename,
                        model_tar_fullpath)
        return ckpt_file

    def _load_graph(self, ckpt_file):
        """
        WARN: internal.

        Load Graph from file.
        Arguments:
            ckpt_file(str): Result from _maybe_download_ckpt()
        Returns:
            graph(tf.Graph): Computational tensorflow graph.
        """
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(ckpt_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def _build_pipeline(self, graph):
        """
        WARN: internal.

        Parse graph into inputs and outputs.
        Arguments:
            graph(tf.Graph)       : Result from _load_graph()
        Returns:
            x(tf.Placeholder)     : NHWC input image tensor (batched)
            y(dict): output(dict) : {'box':A(N,M,4), 'class':A(N,M), 'score':A(N,M), 'num':M=A(N)}
        """
        x = None
        y = {}
        with graph.as_default():
            # convenience alias
            get = lambda s: graph.get_tensor_by_name(s)

            # input
            x = get('image_tensor:0')

            # outputs
            y['box'] = get('detection_boxes:0')
            y['score'] = get('detection_scores:0')
            y['class'] = get('detection_classes:0')
            y['num'] = get('num_detections:0')

            # shape (TODO : fragile)
            rsz = get('Preprocessor/map/while/ResizeImage/resize_images/Squeeze:0')
            shape = tuple(rsz.shape.as_list()[:2]) + (3,)
        return x, y, shape

    def warmup(self):
        """ Allocate resources on the GPU as a warm-up """
        self.__call__(np.zeros(shape=(1,)+self.shape_, dtype=np.uint8))

    def initialize(self):
        """ Create Session and warmup """
        with self.graph_.as_default():
            if self.gpu_ > 0:
                gpu_options = tf.GPUOptions(
                        per_process_gpu_memory_fraction=self.gpu_
                        # TODO : enable allow_growth?
                        )
                config      = tf.ConfigProto(gpu_options=gpu_options)
                self.sess_ = tf.Session(
                        config=config,
                        graph=self.graph_
                        )
            else:
                self.sess_ = tf.Session(
                        config=tf.ConfigProto(device_count={'GPU': 0}),
                        graph=self.graph_
                        )
        self.warmup()

    def run(self, *args, **kwargs):
        return self.sess_.run(*args, **kwargs)

def test_image(img='/tmp/image1.jpg'):
    """
    Simple test script; requires /tmp/image1.jpg
    """
    app = ObjectDetectorTF(model='model')
    img = cv2.imread(img)
    res = app(img)
    msk = (res['score'] > 0.5)

    cls   = res['class'][msk]
    box   = res['box'][msk]
    score = res['score'][msk]
    print('score', score)

    for box_, cls_ in zip(box, cls):
        #ry0,rx0,ry1,rx1 = box_ # relative
        draw_bbox(img, box_, str(cls_))

    cv2.imshow('win', img)
    cv2.waitKey(0)

def test_images(imgdir, recursive=True, is_root=True, shuffle=True, viz=True):
    """
    Simple test script; operating on a directory
    """
    #app = ObjectDetectorTF()
    app = ObjectDetectorTF(gpu=0.5,
            model='model2-drone-300x300',
            #model='model4-drone-300x300',
            #model='model',
            cmap={1:'drone', 2:'person'},
            threshold=0.5,
            threshold2=(0.375, 0.7)
            )

    if is_root and viz:
        cv2.namedWindow('win', cv2.WINDOW_NORMAL)

    fs = os.listdir(imgdir)
    full = False
    if shuffle:
        np.random.shuffle(fs)

    for f in fs:
        f = os.path.join(imgdir, f)
        if os.path.isdir(f):
            if not recursive:
                continue
            if not test_images(f, recursive, is_root=False, shuffle=shuffle, viz=viz):
                break

        img = cv2.imread(f)
        if img is None:
            continue

        #res = app(img[..., ::-1])
        res = app.detect(img, is_bgr=True,
                )

        #msk = (res['score'] > 0.5)
        #if np.count_nonzero(msk) <= 0:
        #    continue

        cls   = res['class']
        box   = res['box']
        score = res['score']
        print('scores', score)
        if viz:
            for box_, cls_, val_ in zip(box, cls, score):
                draw_bbox(img, box_, '{}:{:.2f}%'.format(cls_,int(100.*val_)))
            cv2.imshow('win', img)
            k = cv2.waitKey(0)
            if k in [27, ord('q')]:
                break
    else:
        # went through all images without interruption
        full=True

    if is_root and viz:
        cv2.destroyWindow('win')
        cv2.destroyAllWindows()

    return full

def test_camera():
    """ Simple test srcipt; requires /dev/video0 """
    app = ObjectDetectorTF(gpu=False, cmap={1:'person'})

    cam = cv2.VideoCapture(0)

    fps = []

    while True:
        ret, img = cam.read()
        if not ret:
            print('camera capture failed')
            break
        t0 = time.time()
        res = app(img)
        t1 = time.time()
        fps.append(1.0 / (t1-t0+1e-9))
        msk = (res['score'] > 0.5)

        cls   = res['class'][msk]
        box   = res['box'][msk]
        score = res['score'][msk]
        for box_, cls_ in zip(box, cls):
            #ry0,rx0,ry1,rx1 = box_ # relative
            draw_bbox(img, box_, str(cls_))
        cv2.imshow('win', img)
        k = cv2.waitKey(1)
        if k in [ord('q'), 27]:
            print('quitting...')
            break
        print('average fps: {}'.format( np.mean(fps[-100:])) )

def main():
    #test_image('/tmp/ximg/819.jpg')
    #test_camera()

    #imgdir = '/tmp/simg'
    img_dir = os.path.expanduser(
            #'~/libs/drone-net/image/'
            '/media/ssd/datasets/drones/archive/selfies'
            #'/media/ssd/datasets/drones/data-png/ quadcopter'
            #"/media/ssd/datasets/youtube_box/train/0"
            #'/tmp/selfies'
            )

    #test_images('/media/ssd/datasets/drones/archive/data-png')
    test_images(img_dir, viz=True)
    #test_images("/media/ssd/datasets/coco/raw-data/test2017")

if __name__ == "__main__":
    main()
