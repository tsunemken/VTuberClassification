import argparse
import mlapp
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ResNet152V2')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--source', type=str, default='')
    args = parser.parse_args()
    
    print(args)

    if args.source:
        mlapp.model_setup(args.model)
        img = Image.open(args.source)
        print(mlapp.do_detect(args.model, img, args.img_size))