# 项目文件说明
- models: detect.py 依赖的库 
- utils: detect.py 依赖的库 
- runs/detect: 每次运行detect.py，会将结果存入此文件夹
- YOLOv7
  - images:   数据
  - weight:   权重
- detect.py: 运行前设置好weight和source，结果在runs/detect/exp里找
    > python detect.py --weight YOLOv7/weight/best.pt --source YOLOv7/images/inclusion_3.jpg
