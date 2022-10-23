import mlapp
import utility
import glob
import json
from PIL import Image
import argparse

def training(model, img_size=224, epochs=100, batch_size=8, learning_rate=0.001):
    mlapp.gpu_setup()
    classes = utility.get_classes()
    mlapp.training(model, classes, img_size, epochs, batch_size, learning_rate)

    mlapp.model_setup(model)
    dirs = sorted(glob.glob(utility.TEST_DATA_PATH + '/*'))
    num = 0
    correct = 0

    for dir in dirs:
        dir_name = dir.replace(utility.TEST_DATA_PATH + '/', '')
        files = sorted(glob.glob(dir + '/*'))
        for file in files:
            image = Image.open(file)
            file_name = dir_name + '_' + file.replace(dir + '/', '')
            result = mlapp.do_detect(model, image, img_size)
            result = '{"image":"' + file_name + '", "result":' + result + '}'
            num += 1
            jsonData = json.loads(result)
            if jsonData['result'][0]['label'] == dir_name:
                correct += 1
            else:
                print(result)

    print(str(correct) + ' / ' + str(num))
    utility.save_result(model, correct, num, len(classes))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ResNet152V2')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    
    print(args)

    training(args.model, args.img_size, args.epochs, args.batch_size, args.learning_rate)
