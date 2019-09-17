import utils.training as train_utils
import utils.imgs as img_utils
import numpy as np
import matplotlib

dir = '/home/'
files = ['dbsn_bn_pw0.1_clip_3_1gpu_test_pred.npz', 'baseline_pw0.1_test_mc_pred.npz'] #gpu23

loader = np.load(dir + files[0])
imgs = loader["imgs"]
labs = loader["labs"]
dbsn_preds = loader["preds"]

loader = np.load(dir + files[1])
mcdropout_preds = loader["preds"]

max_ent = 0
for idx in range(imgs.shape[0]):
    #idx = np.random.randint(imgs.shape[0])
    img = imgs[idx]
    lab = labs[idx]

    dbsn_pred = dbsn_preds[idx]
    #print(img.shape, lab.shape, dbsn_pred.shape)
    dbsn_lab = dbsn_pred.argmax(0)
    dbsn_ent = -(np.exp(dbsn_pred) * dbsn_pred).sum(0)
    max_ent = max(max_ent, dbsn_ent.max())

    mcdropout_pred = mcdropout_preds[idx]
    mcdropout_ent = -(np.exp(mcdropout_pred) * mcdropout_pred).sum(0)
    max_ent = max(max_ent, mcdropout_ent.max())

    matplotlib.image.imsave(dir + 'seg_samples/' + str(idx) + '_img.pdf', img.transpose((1, 2, 0)))
    matplotlib.image.imsave(dir + 'seg_samples/' + str(idx) + '_lab.pdf', img_utils.view_annotated(lab, False))
    matplotlib.image.imsave(dir + 'seg_samples/' + str(idx) + '_pred.pdf', img_utils.view_annotated(dbsn_lab, False))

    matplotlib.image.imsave(dir + 'seg_samples/' + str(idx) + '_dbsnent.pdf', dbsn_ent/2.35)
    matplotlib.image.imsave(dir + 'seg_samples/' + str(idx) + '_mcdropoutent.pdf', mcdropout_ent/2.35)

print(max_ent)
