import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as t
from skimage import segmentation
from scipy.ndimage import distance_transform_edt as distance
from torchvision.transforms import Compose


class SaltDataset(Dataset):
    def __init__(self, items, img_height, img_width, transforms=t.ToTensor()):
        super(SaltDataset, self).__init__()
        self.data = items
        self.img_height = img_height
        self.img_width = img_width
        self.transforms = transforms

    def _rle2mask(self, rle):
        # position,length
        m_rle = np.array([int(x) for x in rle.split()]).reshape(-1, 2)
        mask = np.zeros(self.img_height*self.img_width)
        for pos, count in m_rle:
            mask[pos-1:pos+count-1] = 255
        return mask.reshape((self.img_height, self.img_width)).T
            
    def _compute_dtm(self,img_gt, normalize=True, fg=True):
        """
        compute the distance transform map of foreground in binary mask
        input: segmentation, shape = (H, W, C)
        output: the foreground Distance Map (SDM)
        dtm(x) = 0; x in segmentation boundary
                 inf|x-y|; x in segmentation
        """

        fg_dtm = np.zeros(img_gt.shape)

        posmask = img_gt.astype(bool)
        if not fg:
            if posmask.any():
                negmask = 1 - posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = segmentation.find_boundaries(posmask, mode='inner').astype(np.uint8)
                if normalize:
                    fg_dtm = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) + (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                else:
                    fg_dtm = posdis + negdis
                fg_dtm[boundary==1] = 0
        else:
            if posmask.any():
                posdis = distance(posmask)
                boundary = segmentation.find_boundaries(posmask, mode='inner').astype(np.uint8)
                if normalize:
                    fg_dtm = (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                else:
                    fg_dtm = posdis
                fg_dtm[boundary==1] = 0
        return fg_dtm
    
    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.transforms(data)
        data['edge'] = segmentation.find_boundaries(data['mask'],mode='inner')
        data['sdm'] = self._compute_dtm(data['mask'])
        return data

    def __len__(self):
        return len(self.data)

    def add_transform(self, transform):
        self.transforms = Compose([self.transforms, transform])
