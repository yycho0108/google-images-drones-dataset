import os
import numpy as np
import cv2
import shutil

hasher = cv2.img_hash.PHash_create()

def strip_num(s):
    return '.'.join(s.split('.')[1:])[1:]

def dhash(img, h=8):
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (h+1, h))
    diff = img[:, 1:] > img[:, :-1]
    diff = diff.ravel()

    #print 'diff', ''.join([str(e) for e in diff.astype(np.int8)])
    hs = np.power(2, np.arange(len(diff)))*diff
    #print 'hs', hs
    h = hs.sum()
    #print 'hash', h
    return h

def intra_filename(fs):
    fn = [strip_num(f) for f in fs]
    _, idx = np.unique(fn, return_index=True)
    return np.array(fs)[idx]

def inter_filename(ref, fs):
    fn = [strip_num(f) for f in fs]
    msk = np.in1d(fn, ref)
    return fn, fs[~msk]

def genhash(
        droot='/media/ssd/datasets/drones/raw',
        hash_file='/tmp/hs.npy',
        force=False
        ):

    if (not force) and os.path.exists(hash_file):
        print('Hash file {} already exists'.format(hash_file) )
        return

    ds = os.listdir(droot)
    ds = [os.path.join(droot, d) for d in ds]
    fs = [os.listdir(d) for d in ds]
    lfs = [len(f) for f in fs]
    print 'initial sum ', np.sum(lfs)
    itot = np.sum(lfs)

    ## simple filter : by filename
    #fs = [intra_filename(f) for f in fs]
    #ref = []
    #us = []
    #for f in fs:
    #    fn, u = inter_filename(ref, f)
    #    ref.extend(fn)
    #    us.append(u)
    #print np.sum([len(u) for u in us])
    #us1 = us

    #s = set()
    #us = []
    #for f in fs:
    #    u = (set(f) - s) # unique
    #    #s.update(u)
    #    s = s.union(u)
    #    us.append(u)
    #print [len(u) for u in us]
    #print 'set union', len(s)
    #us2 = us

    #print ds[0], ds[1]

    #print set(us2[1]) - set(us1[1])
    #print set(us1[1]) - set(us2[1])
    #print len(us1[1])
    #print len(us2[1])

    #us = [list(u) for u in us]

    hs=[]
    cnt=0
    for d, f in zip(ds, fs):
        b = os.path.basename(d)
        h = {}
        for e in f:
            k = os.path.join(b,e)
            p = os.path.join(d,e)
            #v = dhash(cv2.imread(p))
            img = cv2.imread(p)
            if img is None:
                continue
            v = np.frombuffer(hasher.compute(img), dtype=np.uint64)
            if v is None: #??
                continue
            h[k]=v
            cnt += 1
            if cnt % 100 == 0:
                print '[{}] = {}%'.format(cnt, 100*cnt/float(itot))
        #h = [dhash(cv2.imread(os.path.join(d,e))) for e in f]
        hs.append(h)

    np.save(hash_file, hs)

def putText(img, txt, loc):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 0.4
    fontColor              = (255,0,255)
    lineType               = 1
    cv2.putText(img, txt, 
        loc, 
        font, 
        fontScale,
        fontColor,
        lineType)

def annotate(
        droot='/media/ssd/datasets/drones/raw',
        hash_file='/tmp/hs.npy',
        uout_dir='/tmp/u/',
        show=False
        ):
    hs = np.load(hash_file)
    n = len(hs)

    ixn = np.empty((n,n), dtype=object)

    ks = [sorted(h.keys()) for h in hs]

    msks = [np.ones(len(k), dtype=np.bool) for k in ks]

    for i in range(n):
        ki = np.array(ks[i])
        vi = [hs[i][k] for k in ki]
        if len(vi) <= 0:
            continue

        for j in range(i+1,n):
            kj = np.array(ks[j])
            vj = [hs[j][k] for k in kj]
            if len(vj) <= 0:
                continue

            # opt0: wrong
            #ix = np.intersect1d(vi, vj)
            #ixn[i, j] = (ki[np.in1d(vi, ix)], kj[np.in1d(vj, ix)])

            # opt1 : equality-based (hamming-d=0)
            #idx_i, idx_j = np.argwhere(np.equal(np.reshape(vi, (-1,1)), np.reshape(vj, (1,-1)))).T

            # opt2 : full hamming-based
            xx = np.bitwise_xor(np.reshape(vi, (-1,1)), np.reshape(vj, (1,-1)))
            hd = np.unpackbits(np.frombuffer(xx, dtype=np.uint8)).reshape(len(vi), len(vj), -1).sum(axis=-1, dtype=np.int32)
            idx_i, idx_j = np.argwhere(np.less_equal(hd, 6)).T

            ixn[i,j] = ki[idx_i], kj[idx_j]

            msks[i][idx_i] = False
            msks[j][idx_j] = False

        msk = msks[i]
        print 'unique count : {}/{}={}'.format(msk.sum(), len(vi), (100.0*msk.sum())/len(vi))

    unqs = []
    for i in range(n):
        unqs.extend( np.array(ks[i])[msks[i]] )

    if not os.path.exists(uout_dir):
        os.makedirs(uout_dir)

    idx = 0
    for u in unqs:
        if cv2.imread(os.path.join(droot, u)) is None:
            # image read failure
            continue

        # NOTE: this enforces that the output format is JPEG.
        cv2.imwrite(
                os.path.join(uout_dir, '{:05d}.jpg'.format(idx)))
        #shutil.copyfile(
        #        os.path.join(droot, u),
        #        os.path.join(uout_dir, '{:05d}.jpg'.format(idx))
        #        )
        idx += 1
    print 'idx', idx

    print 'unique', np.sum([np.sum(m) for m in msks])
    print 'unique', len(unqs)
    print 'total',  np.sum([np.size(m) for m in msks])

    if not show:
        return

    cv2.namedWindow('match', cv2.WINDOW_NORMAL)
    for i in range(n):
        for j in range(i+1,n):
            if ixn[i,j] is None:
                continue
            ixi, ixj = ixn[i,j]
            if len(ixi) <= 0 and len(ixj) <= 0:
                continue

            #print ixn[i,j]
            for ii, jj in zip(ixi, ixj):
                imi = cv2.imread( os.path.join(droot, ii))
                imj = cv2.imread( os.path.join(droot, jj))
                hi = hasher.compute(imi)
                hj = hasher.compute(imj)

                vi = np.frombuffer(hi, dtype=np.uint64)
                vj = np.frombuffer(hj, dtype=np.uint64)

                imir = cv2.resize(imi, (320,240))
                imjr = cv2.resize(imj, (320,240))
                viz = np.concatenate([imir, imjr], axis=1)
                putText(viz, str(ii), (0, 200))
                putText(viz, str(jj), (320, 220))
                #print np.intersect1d([vi], [vj])
                #print hi, vi, hs[i][ii]
                #print hj, vj, hs[j][jj]
                cv2.imshow('match', viz)
                k = cv2.waitKey(0)
                if k == ord('q'):
                    return

def main():
    genhash(force=False)
    annotate()

if __name__ == "__main__":
    main()
