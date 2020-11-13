The model I had built has two inputs x,y images (the pair). The backbone is MobileNetv2 cause training time was very large with nearly 17M combinations for permutation invariance and more with augmentation so I had to simplify it to get results quickly, the feature vector of both images are passed into a FC layer with 512 neurons followed by the final FC layer with a sigmoid activation. The accuracy I was able to get it upto was ~60% with 9 epochs of training with CosineAnnealingLR. The usage is pretty straightforward call model(img1,img2) and it should return the sigmoid activation.

References
[1] https://arxiv.org/pdf/1709.01353.pdf
[2] https://openaccess.thecvf.com/content_cvpr_2018/papers_backup/Sung_Learning_to_Compare_CVPR_2018_paper.pdf
[3] https://tspace.library.utoronto.ca/bitstream/1807/43097/3/Liu_Chen_201311_MASc_thesis.pdf
