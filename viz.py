import numpy as np
import os
import cv2

droot = './ann'
fs = sorted(os.listdir(droot))
ls = [len(open(os.path.join(droot, f)).readlines()) for f in fs]
pos_msk = np.greater(ls, 2)
fs = np.array(fs, dtype=str)

fsp = fs[pos_msk]
fsn = fs[~pos_msk]

do_pos = True
#do_pos = False

if do_pos:
    f_lbl = 'ann_pos_lbl.npy'
    f_idx = 'ann_pos_idx.npy'
    fsr = fsp
else:
    f_lbl = 'ann_neg_lbl.npy'
    f_idx = 'ann_neg_idx.npy'
    fsr = fsn

try:
    label = np.load(f_lbl)
except Exception:
    label = np.array( ['x' for _ in range(len(fsr))] )
try:
    i = np.load(f_idx)
except Exception:
    i = 0

cv2.namedWindow('win', cv2.WINDOW_NORMAL)
cv2.resizeWindow('win', 640*2, 480*2)
cv2.moveWindow('win', 320, 0)

while True:
    f = fsr[i]
    f = os.path.join(droot, f)
    l = open(f).readlines()
    imf = l[0][:-1]
    img = cv2.imread(imf)
    box = [[float(e) for e in s.split(' ')] for s in l[2:]]

    h, w = img.shape[:2]
    for b in box:
        bcx,bcy,bw,bh,p = b
        cv2.rectangle(
                img,
                tuple( int(e) for e in (w*(bcx-bw/2.), h*(bcy-bh/2.)) ),
                tuple( int(e) for e in (w*(bcx+bw/2.), h*(bcy+bh/2.)) ),
                (255,0,0),
                int(max(1, 0.02 * (h+w)/2.0))
                )
    cv2.imshow('win', img)
    k = cv2.waitKey(0)

    if k in [ord(e) for e in 'oxs']:
        label[i] = chr(k)
        i += 1
    elif k in [83, ord('k')]:
        i += 1
    elif k in [81, ord('j')]:
        i -= 1
        i = np.clip(i, 0, len(fsr)-1)
    else:
        print 'k', k

    if k in [27, ord('q')]:
        print label[:i]
        print 'o', np.sum(label[:i] == 'o')
        print 'x', np.sum(label[:i] == 'x')
        print 's', np.sum(label[:i] == 's')
        print('current index : {}/{} = {:.02f}%'.format(i, len(fsr), float(i)*100/len(fsr) ))
        break
    
    if i >= len(fsr):
        break

cv2.destroyWindow('win')

np.save(f_idx, i)
np.save(f_lbl, label)
