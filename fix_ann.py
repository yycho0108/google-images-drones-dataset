import numpy as np
import os
import cv2
import shutil

def parse_file(droot, f):
    f = os.path.join(droot, f)
    l = open(f).readlines()
    imf = l[0][:-1]
    img = cv2.imread(imf)
    box = [[float(e) for e in s.split(' ')] for s in l[2:]]

    return imf, img, box

def draw_box(img, b):
    h, w = img.shape[:2]
    #bcx,bcy,bw,bh,p = b
    bcx,bcy,bw,bh = b[:4]
    cv2.rectangle(
            img,
            tuple( int(e) for e in (w*(bcx-bw/2.), h*(bcy-bh/2.)) ),
            tuple( int(e) for e in (w*(bcx+bw/2.), h*(bcy+bh/2.)) ),
            (255,0,0),
            int(max(1, 0.01 * (h+w)/2.0))
            )

def ok_gui(droot, f_lbl, f_idx, fsr):
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
            draw_box(img, b)
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

class BBoxGrabber(object):
    def __init__(self, win):
        cv2.setMouseCallback(win, self.mouse_cb)
        self.data_ = {}
        self.reset()

    def reset(self, shape=None, box=[]):
        self.data_ = {
                'shape' : shape,
                'p0' : None,
                'pc' : None,
                'p1' : None,
                'drawing' : False,
                'box' : box,
                'next' : False,
                'prev' : False
                }

    def mouse_cb(self, event, x, y, flags, param):
        # grab references to the global variables
     
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if (event in [cv2.EVENT_LBUTTONDOWN]):
            if self.data_['drawing'] is False:
                self.data_['p0'] = (x,y)
                self.data_['drawing'] = True
            else:
                self.data_['p1'] = (x,y)
                self.data_['drawing'] = False

                # convert box
                h, w = self.data_['shape'][:2]
                bx0, by0 = self.data_['p0']
                bx1, by1 = self.data_['p1']
                bcx,bcy  = (bx0+bx1)/(2.0*w), (by0+by1)/(2.0*h)
                bw, bh   = abs(bx1-bx0)/(1.0*w), abs(by1-by0)/(1.0*h)

                self.data_['box'].append([bcx,bcy,bw,bh])
        if (event in [cv2.EVENT_MBUTTONDOWN]):
            self.data_['next'] = True
        if (event in [cv2.EVENT_MOUSEWHEEL]):
            if flags > 0:
                self.data_['prev'] = True
            else:
                self.data_['prev'] = False
        elif event == cv2.EVENT_MOUSEMOVE:
            pass
            #self.data_['pc'] = (x,y)

def bbox_gui(droot, f_lbl, fsr,
        do_o=False,
        do_x=True,
        do_s=False
        ):
    label = np.load(f_lbl)
    label = label.astype(str)

    if do_o:
        o_msk = (label == 'o')
        print np.where(o_msk)
        if not os.path.exists('/tmp/o'):
            os.makedirs('/tmp/o')
        for f in fsr[o_msk]:
            # viz
            # imf, img, box = parse_file(droot, f)
            # for b in box:
            #     draw_box(img, b)
            # cv2.imshow('win', img)
            # k = cv2.waitKey(0)
            # if k in [ord('q'), 27]:
            #     break

            # copy
            shutil.copyfile(
                    os.path.join(droot, f),
                    os.path.join('/tmp/o', f)
                    )

    if do_x:
        x_msk = (label == 'x')
        if not os.path.exists('/tmp/x'):
            os.makedirs('/tmp/x')
        for f in fsr[x_msk]:
            ls = open(os.path.join(droot, f)).readlines()[:2]
            open(os.path.join('/tmp/x', f), 'w+').writelines(ls)
            #shutil.copyfile(
            #        os.path.join(droot, f),
            #        os.path.join('/tmp/x', f)
            #        )
        #if not os.path.exists('/tmp/ximg'):
        #    os.makedirs('/tmp/ximg')
        #for (i, f) in enumerate(fsr_x):
        #    l = open(os.path.join(droot, f)).readlines()
        #    imf = l[0][:-1]
        #    shutil.copyfile(imf,
        #            os.path.join('/tmp/ximg', '{}.jpg'.format(i))
        #            )

    if do_s:
        # relabel images labeled 's'
        s_msk = (label == 's')
        print s_msk.sum(), s_msk.size
        print np.where(s_msk)
        fsr_s = fsr[s_msk]

        f_s_idx = './f_s_idx.npy'
        f_s_box = './f_s_box.npy'
        try:
            idx = np.load(f_s_idx)
        except Exception:
            idx = 0

        try:
            boxs = np.load(f_s_box)
        except Exception:
            boxs = [[] for _ in range(len(fsr_s))]

        cv2.namedWindow('win', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('win', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow('win', int(640*1.5), int(480*1.5))
        cv2.moveWindow('win', 320, 0)

        ui = BBoxGrabber('win')

        reload  = True
        img     = None
        overlay = None

        while idx < len(fsr_s):
            # open ... img

            # reload img
            if reload:
                f = fsr_s[idx]
                f = os.path.join(droot, f)
                l = open(f).readlines()
                imf = l[0][:-1]
                img = cv2.imread(imf)
                overlay = img.copy()

                ui.reset(img.shape, boxs[idx])

            if len(ui.data_['box']) > 0:
                overlay = img.copy()
                for b in ui.data_['box']:
                    draw_box(overlay, b)

            if ui.data_['drawing']:
                overlay = img.copy()
                cv2.circle(overlay, ui.data_['p0'],
                        int(max(1, 0.01 * np.mean(img.shape[:2]))),
                        (255,0,0), -1)

            #box = [[float(e) for e in s.split(' ')] for s in l[2:]]
            #for b in box:
            #    draw_box(img, b)

            cv2.imshow('win', cv2.addWeighted(img, 0.5, overlay, 0.5, 0.0) )
            k = cv2.waitKey(1)

            save_box = False
            reload   = True

            if k in [83, ord('k')] or ui.data_['next']:
                # next
                boxs[idx] = ui.data_['box']
                idx += 1
            elif k in [81, ord('j')] or ui.data_['prev']:
                # prev, rolls to back
                boxs[idx] = ui.data_['box']
                idx -= 1
                idx %= len(fsr_s)
            elif k in [27, ord('q')]:
                # quit
                print('current index : {}/{} = {:.02f}%'.format(
                    idx, len(fsr_s), float(idx)*100/len(fsr_s) ))
                break
            elif k in [-1]:
                # no key
                reload = False
                pass
            elif k in [ord('r')]:
                # reset box
                boxs[idx] = []
            else:
                print 'k', k

        cv2.destroyWindow('win')
        np.save(f_s_idx, idx)
        np.save(f_s_box, boxs)

        # write converted results
        if not os.path.exists('/tmp/s'):
            os.makedirs('/tmp/s')

        for (i, (f, box) ) in enumerate( zip(fsr_s, boxs) ):
            l = open(os.path.join(droot, f)).readlines()
            imf = l[0][:-1]
            with open(os.path.join('/tmp/s', f), 'w+') as fout:
                fout.writelines([imf+'\n', 'cx cy w h p\n'])
                for b in box:
                    bcx, bcy, bw, bh = b
                    fout.write('{} {} {} {} {}\n'.format(bcx,bcy,bw,bh,1))
        # write the images
        #if not os.path.exists('/tmp/simg'):
        #    os.makedirs('/tmp/simg')
        #for (i, f) in enumerate(fsr_s):
        #    l = open(os.path.join(droot, f)).readlines()
        #    imf = l[0][:-1]
        #    shutil.copyfile(imf,
        #            os.path.join('/tmp/simg', '{}.jpg'.format(i))
        #            )

def main():
    # basic directory parsing
    droot = './ann'
    fs = sorted(os.listdir(droot))
    ls = [len(open(os.path.join(droot, f)).readlines()) for f in fs]
    pos_msk = np.greater(ls, 2)
    fs = np.array(fs, dtype=str)

    fsp = fs[pos_msk]
    fsn = fs[~pos_msk]

    #do_pos = True
    do_pos = False

    if do_pos:
        f_lbl = 'ann_pos_lbl.npy'
        f_idx = 'ann_pos_idx.npy'
        fsr = fsp
    else:
        f_lbl = 'ann_neg_lbl.npy'
        f_idx = 'ann_neg_idx.npy'
        fsr = fsn

    #ok_gui(droot, f_lbl, f_idx, fsr)
    #bbox_gui(droot, f_lbl, fsr)

    #if not os.path.exists('/tmp/ximg'):
    #    os.makedirs('/tmp/ximg')
    #for (i, f) in enumerate(fsn):
    #    l = open(os.path.join(droot, f)).readlines()
    #    imf = l[0][:-1]
    #    shutil.copyfile(imf,
    #            os.path.join('/tmp/ximg', '{}.jpg'.format(i))
    #            )

if __name__ == "__main__":
    main()
