import numpy as np
import faiss     
from transformers import AutoImageProcessor, ViTModel
import torch

class CosineSimVecDB:
    def __init__(self,vectorDataRoot,img_dir,device):
        self.vectorDataRoot = vectorDataRoot
        self.device = device
        self.img_dir = img_dir
        self.load()
    def load(self):
        self.embeding_model=(ViTModel.from_pretrained("google/vit-base-patch16-384")).to(self.device)
        self.image_preprocessor=AutoImageProcessor.from_pretrained("google/vit-base-patch16-384")
        vectorPath = self.vectorDataRoot + "/vectors.npy"
        paPath = self.vectorDataRoot + "/path.txt"
        vectors_data = np.load(vectorPath)
        pathFileH = open(paPath,"r")
        self.paths = pathFileH.readlines()
        self.paths = [pa[:-1] for pa in self.paths]
        d = vectors_data.shape[1]
        self.quantizer = faiss.IndexFlatIP(d)   # build the index
        nlist = 100
        self.index = faiss.IndexIVFFlat(self.quantizer, d, nlist)
        assert not self.index.is_trained
        self.index.train(vectors_data)
        assert self.index.is_trained
        self.index.add(vectors_data)                  # add vectors to the index
        # print(self.index.ntotal)

    def get_embed(self,pil_rgb_img):
        preprocesssed=self.image_preprocessor(pil_rgb_img, return_tensors='pt').to(self.device)
        with torch.no_grad():
            predict = self.embeding_model(**preprocesssed)['last_hidden_state'][:, 0].detach().cpu().numpy()
            predict = predict/np.linalg.norm(predict,axis=1)
        return predict
    # def get_nearest_img_path(self,pil_rgb_img):
    #     embed_vector=self.get_embed(pil_rgb_img)
    #     nearest_path = self.get_nearest_image_path_vector(embed_vector)
    #     return nearest_path
    def get_nearest_image_path_vector(self,vector):        
        D, I = self.index.search(vector, 5)
        img_paths = [self.img_dir+"/"+self.paths[idx] for idx in I[0]]
        return img_paths
    def get_nearest_img_path(self,pil_rgb_img):
        embed_vector=self.get_embed(pil_rgb_img)
        nearest_path = self.get_nearest_image_path_vector(embed_vector)
        return nearest_path