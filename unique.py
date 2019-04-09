import os
import numpy as np
import cv2
import shutil
import sys

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

def hash_file(f):
    img = cv2.imread(f)
    if img is None:
        return None
    v = np.frombuffer(hasher.compute(img), dtype=np.uint64)
    return v

def genhash(
        droot='/media/ssd/datasets/drones/raw',
        recursive=True,
        level=0
        ):
    droot = os.path.expanduser(droot)

    # collect subdirs / files
    ds = sorted( os.listdir(droot) )
    ds = [os.path.join(droot, d) for d in ds]

    hs = {} # {full_path : hash}
    n = len(ds)
    for i, d in enumerate(ds):
        d = os.path.join(droot, d)
        if os.path.isdir(d):
            # directory
            if recursive:
                hs.update(genhash(d,recursive,level+1))
        else:
            # file
            h = hash_file(d)
            if h is not None:
                hs[d] = h
        if (i % 100) == 0:
            dstr = ('\"' if i>0 else droot) # avoid repeating same string
            print '{}H({}) : {}/{}'.format('\t'*level, dstr, i, n)

    return hs

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

def write_unique(
        hs,
        ref_hs=None,
        uout_dir='/tmp/u/',
        max_hd=5,
        show=False
        ):
    # input
    ks0 = sorted(hs.keys())
    vs0 = [hs[k] for k in ks0] 

    # reference
    # ref organization: (intra;inter)
    ks1 = list(ks0)
    vs1 = list(vs0)
    if ref_hs is not None:
        k_ref = sorted(ref_hs.keys())
        ks1.extend( k_ref )
        vs1.extend( [ref_hs[k] for k in k_ref] )

    # convert to numpy array
    ks0 = np.asarray( ks0 ) # (N,)
    ks1 = np.asarray( ks1 ) # (M,)
    vs0 = np.asarray( vs0 ) # (N,)
    vs1 = np.asarray( vs1 ) # (M,)

    n = len(vs0)
    m = len(vs1)
    #ixn  = np.empty((n,m), dtype=object)

    # hamming distance
    xx = np.bitwise_xor(vs0.ravel()[:,None], vs1.ravel()[None,:]) # (N,M)
    print vs0.shape, vs1.shape, xx.shape
    u8bc = np.unpackbits(np.arange(256,dtype=np.uint8)).reshape(256,-1).sum(axis=-1, dtype=np.uint8)# u8 bit-count
    hd = np.empty(dtype=np.uint8, shape=xx.shape)
    i0 = 0
    cs = 4096 
    xxu8 = xx.view(np.uint8).reshape(n,m,8)
    while True: # try to be good about memory consumption
        print('{}/{}'.format(i0, len(xx)))
        i1 = min(i0+cs, len(xx))
        if i0 == i1:
            break
        np.sum(u8bc[xxu8[i0:i1]], axis=-1, dtype=np.uint8, out=hd[i0:i1])
        i0 = i1

    idx_i = np.arange(n)
    idx_j = np.less_equal(hd, max_hd).argmax(axis=-1) # == "first match"
    dup_msk = np.greater(idx_i, idx_j)
    
    i_dup = idx_i[dup_msk]
    j_dup = idx_j[dup_msk]
    i_unq = idx_i[~dup_msk]

    unqs = ks0[i_unq]
    dup0, dup1 = ks0[i_dup], ks1[j_dup]
    hd_dup = hd[i_dup, j_dup]

    if uout_dir is not None:
        # output unique image files to uout_dir
        if not os.path.exists(uout_dir):
            os.makedirs(uout_dir)
        idx = 0
        for u in unqs:
            img = cv2.imread(u)
            if img is None:
                # image read failure
                continue
            # NOTE: this enforces that the output format is JPEG.
            cv2.imwrite(
                    os.path.join(uout_dir, '{:05d}.jpg'.format(idx)), img)
            # Alternative(fast) :
            #shutil.copyfile(
            #        os.path.join(droot, u),
            #        os.path.join(uout_dir, '{:05d}.jpg'.format(idx))
            #        )
            idx += 1
            if idx%100 == 0:
                print('write {}/{}'.format(idx,n))
    else:
        idx = n

    print 'idx', idx
    print 'total', n
    print 'n_unique', len(unqs)
    print 'n_dup', len(dup0)

    if not show:
        return

    cv2.namedWindow('match', cv2.WINDOW_NORMAL)
    for fi, fj, d in zip(dup0, dup1, hd_dup):
        imi = cv2.imread(fi)
        imj = cv2.imread(fj)

        hi = hasher.compute(imi)
        hj = hasher.compute(imj)

        vi = np.frombuffer(hi, dtype=np.uint64)
        vj = np.frombuffer(hj, dtype=np.uint64)

        imir = cv2.resize(imi, (320,240))
        imjr = cv2.resize(imj, (320,240))
        viz = np.concatenate([imir, imjr], axis=1)
        putText(viz, str(os.path.basename(fi)), (0, 200))
        putText(viz, str(os.path.basename(fj)), (320, 220))

        #print fi
        #print fj
        #print d

        #print np.intersect1d([vi], [vj])
        #print hi, vi, hs[i][ii]
        #print hj, vj, hs[j][jj]
        cv2.imshow('match', viz)
        k = cv2.waitKey(0)
        if k == ord('q'):
            return

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def main():
    # optional: compute reference hash
    #droot    = '/media/ssd/datasets/drones/all'
    #droot    = '/media/ssd/datasets/drones/raw'

    #droot     = '/media/ssd/datasets/drones/data-png/ quadcopter'
    #hash_file = '/tmp/hs-quad-png.npy'
    #ref_hs    = None

    droot     = '/tmp/quad-jpg/'
    hash_file = '/tmp/hs-quad-jpg.npy'
    ref_hs    = np.load('/tmp/hs-quad-png.npy').item()

    #uout_dir  = '/tmp/u'
    uout_dir  = None
    max_hd    = 5

    #droot     = '~/libs/drone-net/image/' # data root
    #hash_file = '/tmp/hs.npy' # hash file
    #ref_hs    = np.load('/tmp/hs-ref.npy').item() # (optional) hash reference
    #uout_dir  = '/tmp/u' # where to write the 'unique' images
    #max_hd    = 5 # max hash distance for image matching

    # determine hash read flag
    read_hash = False
    if os.path.exists(hash_file):
        read_hash = query_yes_no('Load existing hash file at {}?'.format(hash_file))
    # load-or-compute-hash
    if read_hash:
        hs = np.load(hash_file).item() # << necessary to convert to dict
    else:
        hs = genhash(droot=droot)

    # determine hash write flag
    write_hash = True
    if os.path.exists(hash_file):
        write_hash = query_yes_no('Overwrite hash file at {}?'.format(hash_file))
    # maybe-write-hash
    if write_hash:
        np.save(hash_file, hs)
        print('Hash file saved at: {}'.format(hash_file) )

    write_unique(hs, ref_hs=ref_hs,
            uout_dir=uout_dir,
            max_hd=max_hd,
            show=True)

if __name__ == "__main__":
    main()
