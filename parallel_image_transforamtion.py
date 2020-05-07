import numpy as np
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from multiprocessing import Pool

Link_globale_folder = "C:\\Users\\NEW.PC\Desktop\datasets\\2D_images_dataset"
Results_globale_folder = "C:\\Users\\NEW.PC\Desktop\datasets\\2D_images_dataset_FE"
SCALE_PERCENT = 75

folders = ['accordion', 'airplanes', 'anchor', 'ant', 'BACKGROUND_Google', 'barrel', 'bass',
           'beaver', 'binocular', 'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly',
           'camera', 'cannon', 'car_side', 'ceiling_fan', 'cellphone', 'chair', 'chandelier',
           'cougar_body', 'cougar_face', 'crab', 'crayfish', 'crocodile', 'crocodile_head',
           'cup', 'dalmatian', 'dollar_bill', 'dolphin', 'dragonfly', 'electric_guitar',
           'elephant', 'emu', 'euphonium', 'ewer', 'Faces', 'Faces_easy', 'ferry',
           'flamingo', 'flamingo_head', 'garfield', 'gerenuk', 'gramophone', 'grand_piano',
           'hawksbill', 'headphone', 'hedgehog', 'helicopter', 'ibis', 'inline_skate',
           'joshua_tree', 'kangaroo', 'ketch', 'lamp', 'laptop', 'Leopards', 'llama',
           'lobster', 'lotus', 'mandolin', 'mayfly', 'menorah', 'metronome', 'minaret',
           'Motorbikes', 'nautilus', 'octopus', 'okapi', 'pagoda', 'panda', 'pigeon',
           'pizza', 'platypus', 'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone',
           'schooner', 'scissors', 'scorpion', 'sea_horse', 'snoopy', 'soccer_ball',
           'stapler', 'starfish', 'stegosaurus', 'stop_sign', 'strawberry', 'sunflower',
           'tick', 'trilobite', 'umbrella', 'watch', 'water_lilly', 'wheelchair', 'wild_cat',
           'windsor_chair', 'wrench', 'yin_yang']


def open_image(link):
    img = cv2.imread(link, cv2.IMREAD_GRAYSCALE)
    width = int(img.shape[1] * SCALE_PERCENT / 100)
    height = int(img.shape[0] * SCALE_PERCENT / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.IMREAD_GRAYSCALE)
    image = Image.fromarray(resized)
    image = image.convert('1')
    image = np.asarray(image)
    return image


def border_adder(input):
    if type(input) == str:
        image = open_image(input)
    else:
        image = input
    height, width = image.shape[0], image.shape[1]
    X_image = np.zeros((height + 4, width + 4))
    for i in range(height):
        for j in range(width):
            X_image[i + 2, j + 2] = image[i, j]
    for i in range(X_image.shape[0]):
        X_image[i, 0], X_image[i, 1] = X_image[i, 3], X_image[i, 2]
        X_image[i, X_image.shape[1] - 1], X_image[i, X_image.shape[1] - 2] = X_image[i, X_image.shape[1] - 4], X_image[
            i, X_image.shape[1] - 3]
    for i in range(X_image.shape[1]):
        X_image[0, i], X_image[1, i] = X_image[3, i], X_image[2, i]
        X_image[X_image.shape[0] - 1, i], X_image[X_image.shape[0] - 2, i] = X_image[X_image.shape[0] - 4, i], X_image[
            X_image.shape[0] - 3, i]
    return X_image, height, width


from share_maker import X3_X5_filtre, XOR_net


def sharemaker(link):
    image, height, width = border_adder(link)
    sh1 = np.zeros((height, width))
    sh2 = np.zeros((height, width))
    # loop in the image and select X*X neighborgs
    for i in range(2, height + 2):
        for j in range(2, width + 2):
            x, y = X3_X5_filtre(image, i, j)
            input = [x, y]
            result = XOR_net(input)
            if result == 0:
                # To sh1
                sh1[i - 2, j - 2] = image[i, j]
            else:
                # to sh2
                sh2[i - 2, j - 2] = image[i, j]
    return sh1, sh2


# creat folders
def transform(files, files_res):
    for i in tqdm(range(len(files))):
        if files[i] not in files_res:
            link = os.path.join(Link_globale_folder, files[i])
            share1 = sharemaker(link)[0]
            share1_1 = sharemaker(share1)[0]
            share1_1_1 = sharemaker(share1_1)[0]
            plt.imsave(os.path.join(Results_globale_folder, files[i]), share1_1_1, cmap=cm.gray)

def fen(folders_list):
    start =1
    end =10
    for i in range(start,end):
        files_list = os.listdir(os.path.join(Link_globale_folder,folders_list[i]))
        folder_link = os.path.join(Link_globale_folder,folders_list[i])
        res_folder_link =os.path.join(Results_globale_folder,folders_list[i])
        print("{} : {}".format(folders_list[i] , i+1))
        for j in tqdm(range(len(files_list))):
            link = os.path.join(folder_link, files_list[j])
            share1 = sharemaker(link)[0]
            share1_1 = sharemaker(share1)[0]
            share1_1_1 = sharemaker(share1_1)[0]
            plt.imsave(os.path.join(res_folder_link, files_list[j]), share1_1_1, cmap=cm.gray)

#index represent the position in files list THIS PART NEED AUTOMATION
index = 101
files_list = os.listdir(os.path.join(Link_globale_folder, folders[index]))

def process_image(image_name):
    folder_link = os.path.join(Link_globale_folder, folders[index])
    res_folder_link = os.path.join(Results_globale_folder, folders[index])
    link = os.path.join(folder_link,image_name)
    share1 = sharemaker(link)[0]
    share1_1 = sharemaker(share1)[0]
    share1_1_1 = sharemaker(share1_1)[0]
    plt.imsave(os.path.join(res_folder_link, image_name), share1_1_1, cmap=cm.gray)
def main():
    pool = Pool()
    for _ in tqdm(pool.imap_unordered(process_image, files_list), total=len(files_list)):
        pass
    pool.close()

if __name__ == '__main__':
    main()
