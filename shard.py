import tensorflow as tf
import os

def split_tfrecord(tfrecord_path, split_size, out_dir='./records'):
    with tf.Graph().as_default(), tf.Session() as sess:
        ds = tf.data.TFRecordDataset(tfrecord_path).batch(split_size)
        batch = ds.make_one_shot_iterator().get_next()
        part_num = 0
        while True:
            try:
                records = sess.run(batch)
                #part_path = tfrecord_path + '.{:03d}'.format(part_num)
                part_path = os.path.join(out_dir, '{}-{}'.format(
                    os.path.basename(tfrecord_path), part_num))
                with tf.python_io.TFRecordWriter(part_path) as writer:
                    for record in records:
                        writer.write(record)
                part_num += 1
            except tf.errors.OutOfRangeError: break

def main():
    recs = [
            './archive/drone.record',
            './archive/drone-net.record',
            './archive/ximg.record',
            './archive/png.record'
            ]
    for rec in recs:
        split_tfrecord(rec, 256)

if __name__ == '__main__':
    main()
