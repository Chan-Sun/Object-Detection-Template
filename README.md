## Object Detection Project Template
___

This is my template structure for mmdet-based object detection task, mainly learned from [mim-best-practice](https://github.com/open-mmlab/mim-example), [soft-teacher](https://github.com/microsoft/SoftTeacher), and [zhihu blog](https://zhuanlan.zhihu.com/p/437754834).

My goal is building a object detection template project starting from environment construction, which will save much time when facing a new task, working in a new machine or participating in a new competition.

This repo also includes some useful codes writen by myself or slightly modified from mmdet source code, such as train-val split, voc2coco, fewshot data generation, high resolution image slice and some post-analysis code.