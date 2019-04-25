import tensorflow as tf
import numpy as np
import sys
import cv2
import glob

def draw_box(img, box):
    x0, y0, x1, y1 = box[:4]
    h, w = img.shape[:2]
    cv2.rectangle(
            img,
            tuple( int(e) for e in (w*x0, h*y0) ),
            tuple( int(e) for e in (w*x1, h*y1) ),
            (255,0,0),
            int(max(1, 0.01 * (h+w)/2.0))
            )

def main():
    tf.enable_eager_execution()
    #rec_files = glob.glob('./records/*')
    rec_files = glob.glob('../coco/record/*')
    #print rec_files
    np.random.shuffle(rec_files)
    rec_file = rec_files[0]
    print('\trecord file : {}'.format(rec_file))
    rec = tf.data.TFRecordDataset(rec_file)

    #rec = tf.data.TFRecordDataset('./drone-net.record')
    #rec = tf.data.TFRecordDataset('./ximg.record')
    #rec = tf.data.TFRecordDataset('./png.record')
    #rec = tf.data.TFRecordDataset('/tmp/png.record')
    #rec = tf.data.TFRecordDataset('/tmp/ximg2.record')

    proto ={
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

    parse = lambda x : tf.parse_single_example(x, proto)
    dataset = rec.map(parse)
    shuf = dataset.shuffle(buffer_size = 256)
    cv2.namedWindow('win', cv2.WINDOW_NORMAL)
    #print 'tot', sum(1 for _ in tf.python_io.tf_record_iterator('./ximg.record'))
    #print 'tot', sum(1 for _ in tf.python_io.tf_record_iterator('/tmp/ximg.record'))

    for rec in shuf.take(100):
        ks = rec.keys()
        #print [rec[k] for k in ks if k != 'image/encoded']
        print rec['image/filename'].numpy()
        xmin = tf.sparse.to_dense(rec['image/object/bbox/xmin']).numpy()
        xmax = tf.sparse.to_dense(rec['image/object/bbox/xmax']).numpy()
        ymin = tf.sparse.to_dense(rec['image/object/bbox/ymin']).numpy()
        ymax = tf.sparse.to_dense(rec['image/object/bbox/ymax']).numpy()

        jpeg = tf.image.decode_jpeg(rec['image/encoded'])
        img = jpeg.numpy()

        for box in zip(xmin,ymin,xmax,ymax):
            draw_box(img, box)

        cv2.imshow('win', img[...,::-1])
        k = cv2.waitKey(0)

        if k in [ord('q'), 27]:
            break
    cv2.destroyWindow('win')

if __name__ == "__main__":
    main()

#print repr(rec)
#print dataset[0]
