import os
import numpy as np
from PIL import Image
import pylab as pl
import pandas as pd
from tqdm import tqdm
from glob import glob
import torch
import torchvision.transforms as transforms
from dataloader.EyeQ_loader import DatasetGenerator
from networks.densenet_mcf import dense121_mcs
import matplotlib.pyplot as plt


def show_imgs(lt):
    fig, axes = plt.subplots(4, 6, figsize=(20, 20))
    ax = axes.ravel()
    for i in range(24):
        path = lt.loc[i, 'dir']
        q = lt.loc[i, 'quality']
        img = Image.open(path)

        ax[i].imshow(img)
        #     ax[i].set_title(q)
        ax[i].text(50, 350, q, c='white', fontsize=12)
        ax[i].axis('off')
    plt.tight_layout()
    # plt.savefig('/media/zyi/litao/retinal_age_projects/data_proc/fundus_quality_2/qc_samples/ukb_left_samples.png',
    #             dpi=600)
    plt.show()


def qc_main(df):
    lt = df.reset_index(drop=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = dense121_mcs(n_class=3)
    loaded_model = torch.load('DenseNet121_v3_v1.tar')
    model.load_state_dict(loaded_model['state_dict'])
    model.to(device)

    transformList2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    transform_list_val1 = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
        ])
    data_test = DatasetGenerator(df=lt, transform1=transform_list_val1,
                                 transform2=transformList2, n_class=3, set_name='test')
    test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=4,
                                              shuffle=False, num_workers=8, pin_memory=True)

    model.eval()

    results = []
    for epochID, (imagesA, imagesB, imagesC) in enumerate(tqdm(test_loader)):
        imagesA = imagesA.cuda()
        imagesB = imagesB.cuda()
        imagesC = imagesC.cuda()
        _, _, _, _, result_mcs = model(imagesA, imagesB, imagesC)
        result = np.squeeze(result_mcs.data.cpu().numpy())
        if len(result.shape) != 2:
            results.append(result)
        else:
            results.extend(result)

    lt['good-usable-reject'] = results
    label_list = ["Good", "Usable", "Reject"]
    lt['quality'] = lt['good-usable-reject'].apply(lambda x: label_list[np.argmax(x)])
    return lt


def qc_recall_rejects(lt):
    for i, score in enumerate(lt['good-usable-reject']):

        score = score.replace('[', '').replace(']', '').split(' ')
        s = list(filter(None, score))[-1]

        if float(s) < 0.85:
            lt['quality'][i] = 'Usable'

    return lt


if __name__ == '__main__':

    # data_csv = pd.read_csv('/media/zyi/litao/retinal_age_projects/data/young_fundus/test_data_v1.csv')
    # df = pd.DataFrame({'dir': data_csv.filename.str.replace('test_data_v1', '/media/zyi/litao/retinal_age_projects/data/young_fundus/test_data_v1')})   #.str.replace('')
    #
    # if os.path.exists('/media/zyi/litao/retinal_age_projects/data/young_fundus/daxuesheng-qc.csv'):
    #     final = pd.read_csv('/media/zyi/litao/retinal_age_projects/data/young_fundus/daxuesheng-qc.csv')
    # else:
    #     final = qc_main(df)
    #     final.to_csv('/media/zyi/litao/retinal_age_projects/data/young_fundus/daxuesheng-qc.csv', index=False)
    #
    # print(final['quality'].value_counts())
    #
    # final_csv = qc_recall_rejects(final)
    # final_csv.to_csv('/media/zyi/litao/retinal_age_projects/data/young_fundus/daxuesheng-qc_recall.csv', index=False)

    df = pd.read_csv('data/ukb-1/ukb_temp_pred.csv')
    df.filename = 'data/ukb-1/' + df.filename
    # df.dir = df.dir.str.replace('gz_norm', '/media/zyi/litao/retinal_age_projects/data/snap_dataset/gz_norm')
    final = qc_main(df)
    final.to_csv('data/ukb-1/ukb_temp_pred_qc.csv', index=False)
    print(final['quality'].value_counts())



