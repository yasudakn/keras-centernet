#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
from glob import glob
import pandas as pd
import math

from keras_centernet.models.networks.hourglass import HourglassNetwork, normalize_image
from keras_centernet.models.decode import CtDetDecode
from keras_centernet.utils.utils import COCODrawer
from keras_centernet.utils.letterbox import LetterboxTransformer


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--fn', default='assets/demo.jpg', type=str)
  parser.add_argument('--output', default='output', type=str)
  parser.add_argument('--inres', default='512,512', type=str)
  parser.add_argument('--batch-size', default=32, type=int)
  parser.add_argument('--target-classes', default=[0], nargs='*', type=int)
  parser.add_argument('--under-score', default=0.3, type=float)
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
  pred_df = []
  
  batch_loop_count = math.ceil(len(fns) / args.batch_size)
  print('{}/batch_size({})={} batch loop'.format(len(fns), args.batch_size, batch_loop_count))
  
  for j in range(0, batch_loop_count):
    pimgs = np.empty((0, args.inres[0], args.inres[1], 3))
    fnames = []
    last_batch = len(fns) % args.batch_size if (j == batch_loop_count-1) else args.batch_size
    for i in range(0, last_batch):
      fname = next(fns_iter)
      img = cv2.imread(fname)
      letterbox_transformer = LetterboxTransformer(args.inres[0], args.inres[1])
      pimg = letterbox_transformer(img)
#      cv2.imwrite(fname.replace('.jpg', '_lb.jpg'), pimg)
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
        x1, y1, x2, y2 = letterbox_transformer.correct_box(x1, y1, x2, y2)
        if score < args.under_score:
          break
        if 0 == len(args.target_classes) or cl in args.target_classes:
          img = drawer.draw_box(img, x1, y1, x2, y2, score, cl)
          pred_df.append([os.path.basename(fn), x1, y1, x2, y2, score, cl])

      out_fn = os.path.join(args.output, 'ctdet.' + os.path.basename(fn))
      cv2.imwrite(out_fn, img)
      print("Image saved to: %s" % out_fn)

  df = pd.DataFrame(pred_df, columns=['fname', 'x1', 'y1', 'x2', 'y2', 'score', 'cl'])
  df['cl'] = df['cl'].astype(int)
  out_df = os.path.join(args.output, 'ctdet.csv')
  df.to_csv(out_df, index=False)
  print('output csv {} size: {}'.format(out_df, len(pred_df)))

if __name__ == '__main__':
  main()
