import os
from multiprocessing.pool import Pool
from tqdm import tqdm
import requests
import pandas as pd
from PIL import Image

def download(pid, image_list, base_url, save_dir, image_size=(512, 512)):
    colors = ['red', 'green', 'blue']
    for i in tqdm(image_list, postfix=pid):
        img_id = i.split('_', 1)
        for color in colors:
            img_path = img_id[0] + '/' + img_id[1] + '_' + color + '.jpg'
            img_name = i + '_' + color + '.png'
            img_url = base_url + img_path
            if os.path.exists(os.path.join(save_dir, img_name)):
                continue
            else:
                try:
                    r = requests.get(img_url, allow_redirects=True, stream=True)
                    r.raw.decode_content = True
                    im = Image.open(r.raw)
                    im = im.resize(image_size, Image.LANCZOS).convert('L')
                    im.save(os.path.join(save_dir, img_name), 'PNG')
                except:
                    with open('failed_list.txt', 'a') as f:
                        f.write(img_name)
                        f.write('\n')

if __name__ == '__main__':
    process_num = 8
    image_size = (512, 512)
    url = 'http://v18.proteinatlas.org/images/'
    csv_path =  "./HPAv18RBGY_wodpl.csv"
    save_dir = "./external/"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists('failed_list.txt'):
        os.system('touch failed_list.txt')

    print('Parent process %s.' % os.getpid())
    img_list = pd.read_csv(csv_path)['Id']
    list_len = len(img_list)

    p = Pool(process_num)
    for i in range(process_num):
        start = int(i * list_len / process_num)
        end = int((i + 1) * list_len / process_num)
        process_images = img_list[start:end]
        p.apply_async(download, args=(str(i), process_images, url, save_dir, image_size))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
