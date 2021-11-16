# Subcenter-ArcFace-Pytorch
Train and filter data using Subcenter ArcFace model in Pytorch

Paper: [arxiv](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf)

Model:
![alt text](https://camo.githubusercontent.com/c5425fd28d64f6f6de748289ddf8269c9008414bbe824254b6b845c6033f6345/68747470733a2f2f696e7369676874666163652e61692f6173736574732f696d672f6769746875622f73756263656e746572617263666163656672616d65776f726b2e706e67)

#How to use code
For training model modify subcenter-config in config folder. ALso you need to create your API token for neptune logger and put it in new credintials.py file or simply in run_filtration.py.
Markup for your data should be in json like: {'im_path': im_class, ...}

Training: 
'''
python3 run_filtration.py
'''
Filter your data:
Use jupyter notebook in notebooks folder

