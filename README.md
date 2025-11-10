# 敦煌壁画线稿与数字图像特征匹配系统

## 项目描述
本项目实现了基于传统特征匹配方法的敦煌壁画线稿与数字图像对齐系统。

## 环境要求
- Python 3.8+
- 依赖包: 见 requirements.txt

## 使用方法
1. 安装依赖: `pip install -r requirements.txt`
2. 准备数据: 将线稿放入Data/Line/, 壁画放入Data/Pic/
3. 运行程序: `python main.py`
4. 查看结果: 结果保存在results/目录

## 文件说明
- main.py: 主程序入口
- traditional_matcher.py: 特征匹配算法实现
