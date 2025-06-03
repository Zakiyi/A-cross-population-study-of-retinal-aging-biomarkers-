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
    for year in ['2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018']:
        dirs = glob(os.path.join('/media/zyi/litao/retinal_age_projects/data/lingtou/{}-f'.format(year), '*.jpg'))
        df = pd.DataFrame({'dir': dirs})
        final = qc_main(df)
        final.to_csv('/media/zyi/litao/retinal_age_projects/data/lingtou/{}-f-qc.csv'.format(year), index=False)
        print(final['quality'].value_counts())

    csv = ['2010-f-qc.csv', '2011-f-qc.csv', '2012-f-qc.csv', '2013-f-qc.csv', '2014-f-qc.csv', '2015-f-qc.csv',
           '2016-f-qc.csv', '2017-f-qc.csv']

    for c in csv:
        df = pd.read_csv(os.path.join('/media/zyi/litao/retinal_age_projects/data/lingtou/csv_label_qc', c))
        df = qc_recall_rejects(df)
        df.to_csv(os.path.join('/media/zyi/litao/retinal_age_projects/data/lingtou/csv_label_qc_recall', c), index=False)


    qc_csv = []
    for c in csv:
        df = pd.read_csv(os.path.join('/media/zyi/litao/retinal_age_projects/data/lingtou/csv_label_qc_recall', c))
        qc_csv.append(df)

    final_data_csv = pd.concat(qc_csv, ignore_index=True, sort=False)
    final_data_csv.dir = final_data_csv.dir.str.replace('/media/zyi/litao/retinal_age_projects/', '')
    final_data_csv.to_csv('/media/zyi/litao/retinal_age_projects/data/lingtou/csv_label_qc_recall/lingtou_qc.csv', index=False)



