## Notice -- About the data

1. The data here is directly from @Jie Zheng's job, including the original name of the file is not changed. So that you can track back for the source data which located in "毕业论文工作整理-20230623-JieZ\400_论文用的模型和数据等\1_MDOF计算及数据集\1_训练集（KNET1766条）\Datasets".
2. The data here is just part of the raw data of @Jie Zheng's job, in the original datasets there are various data such as 30s, 45s, 60s, 75s, 90s. Go to the directory to see the raw datasets if you want. And don't forget to thanks @Jie Zheng for his job :)
3. Vision Transformer-Base train with the ILSVRC-2012 ImageNet dataset(1k classes and 1.3M Image, image: class = 1300:1). While our first model is using 0.1M ground motion records and only 5 classes(ground motion: damage state = 20:1), means our training dataset is 1/65 of the ViT-Base while ViT-Base only has 78% accuracy.
