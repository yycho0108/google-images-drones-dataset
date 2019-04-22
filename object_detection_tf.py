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
    def __init__(self,
            root='/tmp',
            model='ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03',
            use_gpu=False,
            cmap=None,
            shape=(300,300,3)
            ):
        """
        Arguments:
            root(str): persistent data directory; override to avoid initialization overhead.
            model(str): model name; refer to the [model zoo][2].
            use_gpu(bool): Enable vision processing execution on the GPU.
            cmap(dict): Alternative class definitions; remap known classes for convenience.
            shape(tuple): (WxHx3) shape used for warmup() to pre-allocate tensor.

        Note:
            [2]: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
        """
        # cache arguments
        self.root_  = root
        self.model_ = model
        self.use_gpu_ = use_gpu
        self.shape_ = shape

        # load model
        ckpt_file = self._maybe_download_ckpt()
        self.graph_ = self._load_graph(ckpt_file)
        self.input_, self.output_ = self._build_pipeline(self.graph_)
        self.cmap_ = cmap

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
        outputs = self.run(self.output_, {self.input_:img})
        if self.cmap_ is not None:
            # map to alternative class definitions
            outputs['class'] = [[ str(self.cmap_[x] if (x in self.cmap_) else x) for x in xs] for xs in outputs['class']]
            outputs['class'] = np.array(outputs['class'], dtype=str)
        return outputs

    def detect(self, img,
            is_bgr=True,
            threshold=0.7,
            threshold2=None, 
            ):
        """
        Wrapper around ObjectDetectorTF.__call__() that handles convenience arguments,
        such as two-step detection and channel conversion.
        TODO: currently only works for non-batch style inputs.

        Arguments:
            img(A(H,W,3, uint8)): image to run inference on; batch-mode disabled.
            is_bgr(bool): whether `img` is organized channelwise rgb or bgr.
            threshold(float): minimum bounding-box confidence score to be considered valid.
            threshold2(A(2, float)): parameters weak-strong two-step validation; `None` to disable.
        returns:
            output(dict): {'box':A(M, 4), 'class':A(M), 'score':A(M)}
        """
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
            for c, b in zip(cls[retry_msk], box[retry_msk]):
                sub_img, sup_box = crop_box(img, b, margin=0.25)
                sub_imgs.append(cv2.resize(sub_img, self.shape_[:2][::-1]))
                sup_boxs.append( sup_box )
                sup_clss.append( c )
            sub_imgs = np.stack(sub_imgs, axis=0)
            sub_res = self.__call__(sub_imgs) # TODO : consider `recursive` detection

            new_cls = []
            new_box = []
            new_score = []

            for (spc, spb, sbcs, sbbs, sbss) in zip(
                    sup_clss, sup_boxs,
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
                    new_score.append( sbs )
            new_box = np.reshape(new_box, (-1,4))

            # finalize result
            fin_cls = np.concatenate([cls[good_msk], new_cls], axis=0)
            fin_box = np.concatenate([box[good_msk], new_box], axis=0)
            fin_score = np.concatenate([score[good_msk], new_score], axis=0)

            return {k:v for (k,v) in zip(
                ['class', 'box', 'score'],
                [fin_cls, fin_box, fin_score])}
        # should never reach here
        return None

    def _maybe_download_ckpt(self):
        """
        WARN: internal.

        Check if model file exists; download if not available.
        Returns:
            ckpt_file(str): path to the checkpoint, from which to load graph.
        """
        ckpt_file = os.path.join(self.root_, self.model_,
                'frozen_inference_graph.pb')
        print('ckpt_file', ckpt_file)
        model_tar_basename = '{}.tar.gz'.format(self.model_)
        model_tar_fullpath = os.path.join(self.root_, model_tar_basename)

        if not os.path.exists(ckpt_file):
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
        return ckpt_file

    def _load_graph(self, ckpt_file):
        """
        WARN: internal.

        Load Graph from file.
        Arguments:
            ckpt_file(str): result from _maybe_download_ckpt()
        Returns:
            graph(tf.Graph): computational tensorflow Graph
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
            graph(tf.Graph): result from _load_graph()
        Returns:
            x(tf.Placeholder): NHWC input image tensor (batched)
            y(dict): output(dict): {'box':A(N,M,4), 'class':A(N,M), 'score':A(N,M), 'num':M=A(N)}
        """
        x = None
        y = {}
        with graph.as_default():
            get = lambda s: graph.get_tensor_by_name(s)
            x = get('image_tensor:0')
            y['box'] = get('detection_boxes:0')
            y['score'] = get('detection_scores:0')
            y['class'] = get('detection_classes:0')
            y['num'] = get('num_detections:0')
        return x, y

    def warmup(self):
        """ Allocate resources on the GPU as a warm-up """
        self.__call__(np.zeros(shape=(1,)+self.shape_, dtype=np.uint8))

    def initialize(self):
        """ Create Session and warmup """
        with self.graph_.as_default():
            if self.use_gpu_:
                self.sess_ = tf.Session(graph=self.graph_)
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
    print 'score', score

    for box_, cls_ in zip(box, cls):
        #ry0,rx0,ry1,rx1 = box_ # relative
        draw_bbox(img, box_, str(cls_))

    cv2.imshow('win', img)
    cv2.waitKey(0)

def test_images(imgdir, recursive=True, is_root=True, shuffle=True):
    """
    Simple test script; operating on a directory
    """
    #app = ObjectDetectorTF()
    app = ObjectDetectorTF(use_gpu=False,
            #model='model2-drone-640x640'
            #model='model4-drone-300x300'
            model='model',
            cmap={1:'drone', 2:'person'},
            )

    if is_root:
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
            if not test_images(f, recursive, is_root=False, shuffle=shuffle):
                break

        img = cv2.imread(f)
        if img is None:
            continue

        #res = app(img[..., ::-1])
        res = app.detect(img, is_bgr=True,
                threshold=0.375,
                threshold2=(0.125, 0.5)
                )

        #msk = (res['score'] > 0.5)
        #if np.count_nonzero(msk) <= 0:
        #    continue

        cls   = res['class']
        box   = res['box']
        score = res['score']

        print('scores', score)

        for box_, cls_, val_ in zip(box, cls, score):
            draw_bbox(img, box_, '{}:{:.2f}'.format(cls_,val_))

        cv2.imshow('win', img)
        k = cv2.waitKey(0)
        if k in [27, ord('q')]:
            break
    else:
        # went through all images without interruption
        full=True

    if is_root:
        cv2.destroyWindow('win')
        cv2.destroyAllWindows()

    return full

def test_camera():
    """ Simple test srcipt; requires /dev/video0 """
    app = ObjectDetectorTF(use_gpu=False, cmap={1:'person'})

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
            '~/Repos/drone-net/image'
            #'/media/ssd/datasets/drones/data-png/ quadcopter'
            #"/media/ssd/datasets/youtube_box/train/0"
            #'/tmp/selfies'
            )

    #test_images('/media/ssd/datasets/drones/archive/data-png')
    test_images(img_dir)
    #test_images("/media/ssd/datasets/coco/raw-data/test2017")

if __name__ == "__main__":
    main()
