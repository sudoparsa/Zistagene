# Zistagene

For tissue images at first we run PCA algorithm to reducing dimension of our data which was 256*256*3. With 97% remaining varinace we represent each image with a (1, 1665) numpy array. Then for clustering these arrays we run KMeans clustering algorithm. Here are the resutls:
![image](https://user-images.githubusercontent.com/63104907/188330536-9c2a61c9-7a47-49cf-9c6c-277edd4e2726.png)
![image](https://user-images.githubusercontent.com/63104907/188330541-7ca13266-1739-4f26-90ef-fd230cbb02bf.png)
![image](https://user-images.githubusercontent.com/63104907/188330548-a4d2bbf0-f8be-4414-a14d-1e8a7254709d.png)


# Transfer Learning

For clustering tissue images we used VGG16, VGG19 and ResNet50 pretrained models. By removing top layers, feature maps are used for clustering images.
For clustering we used GMM and KMeans.
Here are results of VGG19 clustering results:

![image](https://user-images.githubusercontent.com/63104907/188331292-5b9c5dfa-0a3a-4fd4-b7b0-74bfa0c2db8d.png)
![image](https://user-images.githubusercontent.com/63104907/188331304-db0eca2d-ea44-4573-afdc-60250bb4a692.png)
![image](https://user-images.githubusercontent.com/63104907/188331312-cf82de07-177a-4687-9d8c-83dc3c12fac0.png)
![image](https://user-images.githubusercontent.com/63104907/188331300-9768c375-06e9-468f-8ae1-c2eb56e01bba.png)
![image](https://user-images.githubusercontent.com/63104907/188331319-10a746eb-6711-40a0-b698-a6b7fa27b179.png)

# WBC

Results of white-blood-cell detection model on our data with diffrent zooms:
![image](https://user-images.githubusercontent.com/63104907/188331431-0eb42054-17bc-4093-ab6c-2c8dc7176f3f.png)
![image](https://user-images.githubusercontent.com/63104907/188331450-7932e601-ed0b-4d5f-8e14-8306b5656f07.png)

