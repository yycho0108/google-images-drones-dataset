import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import cv2
import time

from utils import draw_bbox

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
            img(A(N?,W,H,3)): image to run inference on; may optionally be batched.
        Returns:
            output(dict): {'box':A(N,M,4), 'class':A(N,M), 'score':A(N,M)}
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

def test_image():
    """
    Simple test script; requires /tmp/image1.jpg
    """
    app = ObjectDetectorTF()
    img = cv2.imread('/tmp/image1.jpg')
    h,w = img.shape[:2]
    res = app(img)
    msk = (res['score'] > 0.5)

    cls   = res['class'][msk]
    box   = res['box'][msk]
    score = res['score'][msk]

    for box_, cls_ in zip(box, cls):
        #ry0,rx0,ry1,rx1 = box_ # relative
        draw_bbox(img, box_, str(cls_))

    cv2.imshow('win', img)
    cv2.waitKey(0)

def test_images():
    """
    Simple test script; requires /tmp/image1.jpg
    """
    #app = ObjectDetectorTF()
    app = ObjectDetectorTF(model='model2')

    #imgdir = '/tmp/simg'
    imgdir = os.path.expanduser(
            #'~/libs/drone-net/image'
            '/media/ssd/datasets/drones/data-png/ quadcopter'
            )

    cv2.namedWindow('win', cv2.WINDOW_NORMAL)

    fs = os.listdir(imgdir)
    np.random.shuffle(fs)
    
    for f in fs:
        f = os.path.join(imgdir, f)
        img = cv2.imread(f)
        if img is None:
            continue

        h,w = img.shape[:2]
        res = app(img)
        msk = (res['score'] > 0.5)

        cls   = res['class'][msk]
        box   = res['box'][msk]
        score = res['score'][msk]

        print('scores', score)

        for box_, cls_ in zip(box, cls):
            #ry0,rx0,ry1,rx1 = box_ # relative
            draw_bbox(img, box_, str(cls_))

        cv2.imshow('win', img)
        k = cv2.waitKey(0)
        if k in [27, ord('q')]:
            break

    cv2.destroyWindow('win')

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
        h,w = img.shape[:2]
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
    #test_image()
    #test_camera()
    test_images()

if __name__ == "__main__":
    main()
