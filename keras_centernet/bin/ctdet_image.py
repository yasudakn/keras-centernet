#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
from glob import glob

from keras_centernet.models.networks.hourglass import HourglassNetwork, normalize_image
from keras_centernet.models.decode import CtDetDecode
from keras_centernet.utils.utils import COCODrawer
from keras_centernet.utils.letterbox import LetterboxTransformer


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--fn', default='assets/demo.jpg', type=str)
  parser.add_argument('--output', default='output', type=str)
  parser.add_argument('--inres', default='512,512', type=str)
  parser.add_argument('--batch-size', default=32)
  parser.add_argument('--classes', default='', nargs='*')
  args, _ = parser.parse_known_args()
  args.inres = tuple(int(x) for x in args.inres.split(','))
  os.makedirs(args.output, exist_ok=True)
  kwargs = {
    'num_stacks': 2,
    'cnv_dim': 256,
    'weights': 'ctdet_coco',
    'inres': args.inres,
  }
  heads = {
    'hm': 80,  # 3
    'reg': 2,  # 4
    'wh': 2  # 5
  }
  model = HourglassNetwork(heads=heads, **kwargs)
  model = CtDetDecode(model)
  drawer = COCODrawer()
  fns = sorted(glob(args.fn))
  fns_iter = iter(fns)
  
  print('{}/{} {}'.format(len(fns), args.batch_size, int(len(fns) / args.batch_size)))
  
  for j in range(0, int(len(fns) / args.batch_size)):
    pimgs = np.empty((0, args.inres[0], args.inres[1], 3))
    fnames = []
    for i in range(0, args.batch_size):
      fname = next(fns_iter)
      img = cv2.imread(fname)
      letterbox_transformer = LetterboxTransformer(args.inres[0], args.inres[1])
      pimg = letterbox_transformer(img)
      pimg = normalize_image(pimg)
      pimg = np.expand_dims(pimg, 0)
      pimgs = np.append(pimgs, pimg, axis=0)
      fnames.append(fname)
    print(pimgs.shape)

    predicts = model.predict(pimgs)
    print(predicts.shape)

    for p, fn in zip(predicts, fnames):
      img = cv2.imread(fn)
      for d in p:
        x1, y1, x2, y2, score, cl = d
        if 0 < len(args.classes) and cl in args.classes:
          if score < 0.3:
            break
          x1, y1, x2, y2 = letterbox_transformer.correct_box(x1, y1, x2, y2)
          img = drawer.draw_box(img, x1, y1, x2, y2, cl)

      out_fn = os.path.join(args.output, 'ctdet.' + os.path.basename(fn))
      cv2.imwrite(out_fn, img)
      print("Image saved to: %s" % out_fn)


if __name__ == '__main__':
  main()
