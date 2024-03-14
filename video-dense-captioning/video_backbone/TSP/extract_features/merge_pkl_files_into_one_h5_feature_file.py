from __future__ import division, print_function

import argparse
import pickle as pkl
import h5py
import glob
import os

from tqdm import tqdm


def main(args):
    print(args)
    compression_flags = dict(compression='gzip', compression_opts=9)
    filenames = glob.glob(os.path.join(args.features_folder, '*.pkl'))
    print(f'Number of pkl files: {len(filenames)}')

    output = h5py.File(args.output_h5, 'w')
    for f in tqdm(filenames):
        video_name = os.path.basename(f).split('.pkl')[0]
        with open(f, 'rb') as fobj:
            data = pkl.load(fobj)
        output.create_dataset(video_name, data=data, chunks=True, **compression_flags)

    output.close()
    print(f'The h5 feature file is saved to {args.output_h5}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge the feature pkl files of different videos into one '
                                                 'h5 feature file mapping video name to feature tensor.')

    parser.add_argument('--features-folder', required=True, type=str,
                      help='Path to the folder containing the pkl feature files')
    parser.add_argument('--output-h5', required=True, type=str,
                      help='Where to save the combined metadata CSV file')

    args = parser.parse_args()

    main(args)
