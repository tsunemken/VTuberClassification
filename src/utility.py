from genericpath import isfile
import shutil
import subprocess
import glob
import json
import os
import datetime
import csv
from PIL import Image

CLASSES_PATH = '/root/project/VTuberClassification/workdirectory/model/classes.txt'
TEST_DATA_PATH = '/root/project/VTuberClassification/workdirectory/dataset/test'
DATA_PATH = '/root/project/VTuberClassification/workdirectory/image'
DATASET_PATH = '/root/project/VTuberClassification/workdirectory/dataset'
MODEL_PATH = '/root/project/VTuberClassification/workdirectory/model'
RESULT_PATH = '/root/project/VTuberClassification/workdirectory/result'
RESULT_JSON_PATH = '/root/project/VTuberClassification/workdirectory/result/result.json'

MODEL_DENSENET121 = 'DenseNet121'
MODEL_DENSENET169 = 'DenseNet169'
MODEL_DENSENET201 = 'DenseNet201'
MODEL_EFFICIENTNETB0 = 'EfficientNetB0'
MODEL_EFFICIENTNETB1 = 'EfficientNetB1'
MODEL_EFFICIENTNETB2 = 'EfficientNetB2'
MODEL_EFFICIENTNETB3 = 'EfficientNetB3'
MODEL_EFFICIENTNETB4 = 'EfficientNetB4'
MODEL_EFFICIENTNETB5 = 'EfficientNetB5'
MODEL_EFFICIENTNETB6 = 'EfficientNetB6'
MODEL_EFFICIENTNETB7 = 'EfficientNetB7'
MODEL_INCEPTIONRESNETV2 = 'InceptionResNetV2'
MODEL_INCEPTIONV3 = 'InceptionV3'
MODEL_MOVBILENET = 'MobileNet'
MODEL_MOVBILENETV2 = 'MobileNetV2'
MODEL_MOVBILENETV3LARGE = 'MobileNetV3Large'
MODEL_MOVBILENETV3SMALL = 'MobileNetV3Small'
MODEL_NASNETLARGE = 'NASNetLarge'
MODEL_NASNETMOBILE = 'NASNetMobile'
MODEL_RESNET50 = 'ResNet50'
MODEL_RESNET50V2 = 'ResNet50V2'
MODEL_RESNET101 = 'ResNet101'
MODEL_RESNET101V2 = 'ResNet101V2'
MODEL_RESNET152 = 'ResNet152'
MODEL_RESNET152V2 = 'ResNet152V2'
MODEL_VGG16 = 'VGG16'
MODEL_VGG19 = 'VGG19'
MODEL_XCEPTION = 'Xception'

def remove_my_dataset():
    # remove mydataset folder
    try:
        shutil.rmtree(DATASET_PATH)
    except OSError as e:
        print("Error: %s : %s" % (DATASET_PATH, e.strerror))

def resize(im, desired_size=224):
    old_size = im.size 

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = im.resize(new_size, Image.ANTIALIAS)

    new_im = Image.new('RGB', (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))

    return new_im

def remove_transparency(im, bg_colour=(255, 255, 255)):    
    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        bg = Image.new("RGB", im.size, bg_colour)
        bg.paste(im, mask=im.split()[3])
        return bg
    else:
        return im

def get_gpu_memory_usage():
    """
    Grab nvidia-smi output and return a dictionary of the memory usage.
    """
    data = {}

    try:
        p = subprocess.Popen(['nvidia-smi -q'], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # output, err = p.communicate()
        countLine = 0
        for line in p.stdout:
            if b"FB Memory Usage" in line:
                # print(line.decode('utf8'))
                countLine += 1
                continue
            if countLine > 0 and countLine <= 3:
                line = line.decode('utf8')
                temp_arr = [x.strip() for x in line.split(':')]
                # print(tempArr)
                data[temp_arr[0]] = temp_arr[1]
                countLine += 1
                if (countLine > 3):
                    break
                else:
                    continue

    except (OSError, ValueError) as e:
        pass
    return data

def get_classes():
    classes = []
    files = sorted(glob.glob(DATA_PATH + '/*'))
    for file in files:
        file = file.replace(DATA_PATH + '/', '')
        classes.append(file)

    return classes

def save_result(model_name, correct, num, classes_size):
    result = dict()

    now = datetime.datetime.now()
    resule_timestamp = now.strftime('%Y%m%d_%H%M%S')

    if os.path.isfile(RESULT_JSON_PATH):
        json_file = open(RESULT_JSON_PATH, "r")
        result = json.load(json_file)
        json_file.close()

    result[model_name] = {'score':correct / num, 'timestamp':resule_timestamp, 'classes_size':classes_size}

    json_file = open(RESULT_JSON_PATH, "w")
    json.dump(result, json_file, indent=4)
    json_file.close()

    result = dict()

    json_file_path = os.path.join(RESULT_PATH, 'result-' + model_name + '.json')
    if os.path.isfile(json_file_path):
        json_file = open(json_file_path, "r")
        result = json.load(json_file)
        json_file.close()

    result[resule_timestamp] = correct / num

    json_file = open(json_file_path, "w")
    json.dump(result, json_file, indent=4)
    json_file.close()

def save_test_result(model_name, result_list):
    json_file = os.path.join(RESULT_PATH, 'test-' + model_name + '.json')
    result = {'result_list': result_list}
    
    json_file = open(json_file, "w")
    json.dump(result, json_file, indent=4)
    json_file.close()

def save_inference_csv(model_name, result_list):
    csv_file = os.path.join(RESULT_PATH, 'inference-' + model_name + '.csv')

    with open(csv_file, "w", newline="\n") as f:
        writer = csv.writer(f,delimiter=",")
        writer.writerows(result_list)

def save_inference_json(model_name, result_list):
    json_file = os.path.join(RESULT_PATH, 'inference-' + model_name + '.json')
    result = {'result_list': result_list}
    
    json_file = open(json_file, "w")
    json.dump(result, json_file, indent=4)
    json_file.close()
