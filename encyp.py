import cv2 as cv
import numpy as np
from numpy import e, random
from matplotlib import pyplot as plt
class S_box():
    def __init__(self):
        self.s_box={}
        self.reverse_s_box={}
        self.pool=list(range(256))
    def generateBox(self):
        for i in range(256):
            random_val=np.random.randint(len(self.pool))
            self.s_box[i]=self.pool[random_val]
            self.reverse_s_box[self.pool[random_val]]=i
            self.pool.remove(self.pool[random_val])
    def reverseBox(self):
        for i in self.s_box:
            self.reverse_s_box[i[1]+i[0]]=self.s_box[i]
    def getSbox(self):
        print(self.s_box)
    def getRSbox(self):
        print(self.reverse_s_box)
class EncryptionModel():
    def __init__(self,img_path) :
        self.img=cv.imread(img_path,1)
        self.img_path=img_path
        self.image_shape=self.img.shape
        self.max_ind=list()
        self.chanel_array=list()
        self.init_array=list()
        self.random_array=list()
        self.XORed_img=list()
        self.sbox=S_box()
        self.sbox.generateBox()
    def getImg(self):
        print(self.img)
    def getImgPath(self):
        print(self.img_path)
    def getImgSize(self):
        print(self.image_size)
    def getMaxIndArray(self):
        print(self.max_ind)
    def getChanelArray(self):
        print(self.chanel_array)
    def getXORedImg(self):
        print(self.XORed_img)
    def getRandArray(self):
        print(self.random_array)
    def generateRandomArray(self):
        for i in range(self.image_shape[1]):
            self.random_array.append(np.random.randint(0,256))
        self.init_array=self.random_array
    def max_pixCha(self):
        max_index=list()
        for i in self.img:
            max_rowInd=list()
            for j in i:
                max_rowInd.append(np.argmax(j))
            max_index.append(max_rowInd)
        self.max_ind=max_index
    def applyPHMM(self):
        ind=self.max_ind
        maxProbValue=3
        bgr=[0,0,0]#blue green red
        for i in range(len(ind[0])):
            if i==0:
                for j in ind:
                    if(j[i]==0):
                        bgr[0]+=1
                    elif(j[i]==1):
                        bgr[1]+=1
                    else:
                        bgr[2]+=1
                maxProbValue=bgr.index(max(bgr))
                self.chanel_array.append(maxProbValue)
                bgr=[0,0,0]
            else:
                for j in ind:
                    if (j[i-1]==maxProbValue):
                        if(j[i]==0):
                            bgr[0]+=1
                        elif(j[i]==1):
                            bgr[1]+=1
                        else:
                            bgr[2]+=1
                maxProbValue=bgr.index(max(bgr))
                self.chanel_array.append(maxProbValue)
                bgr=[0,0,0]
    def applyEncyp(self,encryptedImageName):
        for i in range(self.image_shape[0]):
            for j in range(self.image_shape[1]):
                b=self.img[i,j,0]^self.random_array[j]
                b=self.sbox.s_box[b]
                g=self.img[i,j,1]^self.random_array[j]
                g=self.sbox.s_box[g]
                r=self.img[i,j,2]^self.random_array[j]
                r=self.sbox.s_box[r]
                if self.chanel_array[j]==0:
                    self.random_array[j]=self.img[i,j,0]^self.random_array[j]
                    self.img[i,j,0]=b
                    self.img[i,j,1]=g
                    self.img[i,j,2]=r
                elif self.chanel_array[j]==1:
                    self.random_array[j]=self.img[i,j,1]^self.random_array[j]
                    self.img[i,j,0]=b
                    self.img[i,j,1]=g
                    self.img[i,j,2]=r
                else:
                    self.random_array[j]=self.img[i,j,2]^self.random_array[j]
                    self.img[i,j,0]=b
                    self.img[i,j,1]=g
                    self.img[i,j,2]=r
        cv.imwrite(encryptedImageName,self.img)
        #plt.hist(self.img.ravel(),256,[0,256]); plt.show()
        #cv.imshow('sifreli',self.img)
        #cv.waitKey(0)
    def applyDecyp(self,dencryptedImageName):
        for i in range(self.image_shape[0]):
            for j in range(self.image_shape[1]):
                b=self.sbox.reverse_s_box[self.img[i,j,0]]
                self.img[i,j,0]=b^self.init_array[j]
                g=self.sbox.reverse_s_box[self.img[i,j,1]]
                self.img[i,j,1]=g^self.init_array[j]
                r=self.sbox.reverse_s_box[self.img[i,j,2]]
                self.img[i,j,2]=r^self.init_array[j]
                if self.chanel_array[j]==0:
                    self.init_array[j]=b
                elif self.chanel_array[j]==1:
                    self.init_array[j]=g
                else:
                    self.init_array[j]=r
        cv.imwrite(dencryptedImageName,self.img)
        #cv.imshow('sifreli',self.img)
        #cv.waitKey(0)
        
    """
    def applyEncyp(self):
        for i in range(self.image_shape[0]):
            for j in range(self.image_shape[1]):
                if self.chanel_array[j]==0:
                    b=self.img[i,j,0]^self.random_array[j]
                    b=self.sbox.s_box[b]
                    g=self.img[i,j,1]^self.random_array[j]
                    g=self.sbox.s_box[g]
                    r=self.img[i,j,2]^self.random_array[j]
                    r=self.sbox.s_box[r]
                    self.random_array[j]=self.img[i,j,0]^self.random_array[j]
                    self.img[i,j,0]=b
                    self.img[i,j,1]=g
                    self.img[i,j,2]=r
                elif self.chanel_array[j]==1:
                    b=self.img[i,j,0]^self.random_array[j]
                    b=self.sbox.s_box[b]
                    g=self.img[i,j,1]^self.random_array[j]
                    g=self.sbox.s_box[g]
                    r=self.img[i,j,2]^self.random_array[j]
                    r=self.sbox.s_box[r]
                    self.random_array[j]=self.img[i,j,1]^self.random_array[j]
                    self.img[i,j,0]=b
                    self.img[i,j,1]=g
                    self.img[i,j,2]=r
                else:
                    b=self.img[i,j,0]^self.random_array[j]
                    b=self.sbox.s_box[b]
                    g=self.img[i,j,1]^self.random_array[j]
                    g=self.sbox.s_box[g]
                    r=self.img[i,j,2]^self.random_array[j]
                    r=self.sbox.s_box[r]
                    self.random_array[j]=self.img[i,j,2]^self.random_array[j]
                    self.img[i,j,0]=b
                    self.img[i,j,1]=g
                    self.img[i,j,2]=r
        cv.imshow('sifreli',self.img)
        cv.waitKey(0)
    
    def applyDecyp(self):
        self
        for i in range(self.image_shape[0]):
            for j in range(self.image_shape[1]):
                self.img[i,j,0]=self.sbox.reverse_s_box[self.img[i,j,0]]
                self.img[i,j,1]=self.sbox.reverse_s_box[self.img[i,j,1]]
                self.img[i,j,2]=self.sbox.reverse_s_box[self.img[i,j,2]]
        for i in range(self.image_shape[0]):
            for j in range(self.image_shape[1]):
                if self.chanel_array[j]==0:
                    t=self.init_array[j]
                    r=self.img[i,j,2]^t
                    g=self.img[i,j,1]^t
                    self.init_array[j]=self.img[i,j,0]^t
                    self.img[i,j,2]=r
                    self.img[i,j,1]=g
                    self.img[i,j,0]=self.init_array[j]
                    self.init_array[j]=t^self.init_array[j]
                elif self.chanel_array[j]==1:
                    t=self.init_array[j]
                    b=self.img[i,j,0]^t
                    r=self.img[i,j,2]^t
                    self.init_array[j]=self.img[i,j,1]^t
                    self.img[i,j,0]=b
                    self.img[i,j,2]=r
                    self.img[i,j,1]=self.init_array[j]
                    self.init_array[j]=t^self.init_array[j]
                else:
                    t=self.init_array[j]
                    b=self.img[i,j,0]^t
                    g=self.img[i,j,1]^t
                    self.init_array[j]=self.img[i,j,2]^t
                    self.img[i,j,0]=b
                    self.img[i,j,1]=g
                    self.img[i,j,2]=self.init_array[j]
                    self.init_array[j]=t^self.init_array[j]
        cv.imshow('cozum',self.img)
        cv.waitKey(0)
    """
encyp=EncryptionModel('uydu4.jpg')
encyp.generateRandomArray()
encyp.max_pixCha()
encyp.applyPHMM()
encyp.applyEncyp("ResultDir/uydu4Encrypted.jpg")
encyp.applyDecyp("ResultDir/uydu4Dencrypted.jpg")
