import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import pandas as pd

class TraditionalFeatureMatcher:
    def __init__(self):
        # 初始化SIFT检测器 - 调整参数以适应线稿
        self.sift = cv2.SIFT_create(
            nfeatures=2000,           # 增加特征点数量
            contrastThreshold=0.01,    # 降低对比度阈值
            edgeThreshold=5           # 降低边缘阈值
        )
        # FLANN匹配器参数
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # 初始化属性
        self.H = None
        self.inlier_matches = []
        self.inlier_ratio = 0
    
    def preprocess_images(self, line_img, mural_img):
        """预处理图像，增强特征提取"""
        # 统一图像尺寸
        target_size = (800, 600)
        
        line_img = cv2.resize(line_img, target_size)
        mural_img = cv2.resize(mural_img, target_size)
        
        # 对线稿进行二值化处理
        _, line_binary = cv2.threshold(line_img, 127, 255, cv2.THRESH_BINARY)
        
        # 对壁画进行边缘检测，使其更接近线稿风格
        mural_edges = cv2.Canny(mural_img, 50, 150)
        
        # 对壁画进行对比度增强
        mural_enhanced = cv2.equalizeHist(mural_img)
        
        return line_binary, mural_enhanced, mural_edges
    
    def load_images(self, line_path, mural_path):
        """加载线稿和壁画图像"""
        self.line_img = cv2.imread(line_path, cv2.IMREAD_GRAYSCALE)
        self.mural_img = cv2.imread(mural_path, cv2.IMREAD_GRAYSCALE)
        
        if self.line_img is None or self.mural_img is None:
            raise ValueError(f"无法加载图像，请检查路径:\n线稿: {line_path}\n壁画: {mural_path}")
        
        print(f"原始尺寸 - 线稿: {self.line_img.shape}, 壁画: {self.mural_img.shape}")
        
        # 预处理图像
        self.line_processed, self.mural_enhanced, self.mural_edges = self.preprocess_images(
            self.line_img, self.mural_img
        )
        print(f"处理后尺寸 - 线稿: {self.line_processed.shape}, 壁画: {self.mural_enhanced.shape}")
        
        return self.line_processed, self.mural_enhanced
    
    def extract_features(self):
        """提取SIFT特征 - 尝试多种策略"""
        # 策略1: 线稿 vs 增强后的壁画
        self.kp1, self.des1 = self.sift.detectAndCompute(self.line_processed, None)
        self.kp2_enhanced, self.des2_enhanced = self.sift.detectAndCompute(self.mural_enhanced, None)
        
        # 策略2: 线稿 vs 边缘检测后的壁画
        self.kp2_edges, self.des2_edges = self.sift.detectAndCompute(self.mural_edges, None)
        
        print(f"线稿特征点数量: {len(self.kp1)}")
        print(f"增强壁画特征点数量: {len(self.kp2_enhanced)}")
        print(f"边缘壁画特征点数量: {len(self.kp2_edges)}")
        
        # 选择特征点较多的壁画版本
        if len(self.kp2_enhanced) >= len(self.kp2_edges):
            self.kp2 = self.kp2_enhanced
            self.des2 = self.des2_enhanced
            self.mural_for_matching = self.mural_enhanced
            print("使用增强版壁画进行匹配")
        else:
            self.kp2 = self.kp2_edges
            self.des2 = self.des2_edges
            self.mural_for_matching = self.mural_edges
            print("使用边缘版壁画进行匹配")
        
        return self.kp1, self.des1, self.kp2, self.des2
    
    def match_features(self, ratio_threshold=0.8):  # 提高阈值以获取更多匹配
        """使用KNN匹配特征点"""
        if self.des1 is None or self.des2 is None:
            print("错误：无法提取特征描述符")
            self.good_matches = []
            return self.good_matches
        
        # 确保描述符为float32类型
        if self.des1.dtype != np.float32:
            self.des1 = self.des1.astype(np.float32)
        if self.des2.dtype != np.float32:
            self.des2 = self.des2.astype(np.float32)
        
        try:
            matches = self.flann.knnMatch(self.des1, self.des2, k=2)
            
            # 应用Lowe's ratio test
            self.good_matches = []
            for m, n in matches:
                if len(matches) > 0 and m.distance < ratio_threshold * n.distance:
                    self.good_matches.append(m)
            
            print(f"初始匹配数量: {len(matches)}")
            print(f"经过ratio test后的匹配数量: {len(self.good_matches)}")
            
            # 如果匹配点太少，尝试更宽松的阈值
            if len(self.good_matches) < 10:
                print("匹配点过少，尝试更宽松的ratio阈值: 0.9")
                self.good_matches = []
                for m, n in matches:
                    if m.distance < 0.9 * n.distance:
                        self.good_matches.append(m)
                print(f"宽松阈值后的匹配数量: {len(self.good_matches)}")
            
        except Exception as e:
            print(f"FLANN匹配失败: {e}")
            # 尝试使用暴力匹配
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(self.des1, self.des2, k=2)
            
            self.good_matches = []
            for m, n in matches:
                if m.distance < ratio_threshold * n.distance:
                    self.good_matches.append(m)
            
            print(f"暴力匹配 - 初始匹配数量: {len(matches)}")
            print(f"暴力匹配 - 经过ratio test后的匹配数量: {len(self.good_matches)}")
        
        return self.good_matches
    
    def filter_matches_ransac(self, reproj_threshold=5.0):
        """使用RANSAC过滤匹配点"""
        if len(self.good_matches) < 4:
            print("匹配点太少，无法进行RANSAC过滤")
            self.H = None
            self.inlier_matches = []
            self.inlier_ratio = 0
            return None, []
        
        # 准备点对
        src_pts = np.float32([self.kp1[m.queryIdx].pt for m in self.good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.kp2[m.trainIdx].pt for m in self.good_matches]).reshape(-1, 1, 2)
        
        try:
            # 使用RANSAC计算单应性矩阵
            self.H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_threshold)
            
            if mask is None:
                print("RANSAC失败，无法计算单应性矩阵")
                self.H = None
                self.inlier_matches = []
                self.inlier_ratio = 0
                return None, []
            
            # 提取内点
            self.inlier_matches = [self.good_matches[i] for i in range(len(mask)) if mask[i] == 1]
            self.inlier_ratio = len(self.inlier_matches) / len(self.good_matches) if len(self.good_matches) > 0 else 0
            
            print(f"RANSAC内点数量: {len(self.inlier_matches)}")
            print(f"内点比例: {self.inlier_ratio:.3f}")
            
            return self.H, self.inlier_matches
            
        except Exception as e:
            print(f"RANSAC计算失败: {e}")
            self.H = None
            self.inlier_matches = []
            self.inlier_ratio = 0
            return None, []
    
    def calculate_alignment_error(self):
        """计算对齐误差"""
        if self.H is None or len(self.inlier_matches) == 0:
            print("无法计算对齐误差：单应性矩阵或内点为空")
            return float('inf')
        
        src_pts = np.float32([self.kp1[m.queryIdx].pt for m in self.inlier_matches])
        dst_pts = np.float32([self.kp2[m.trainIdx].pt for m in self.inlier_matches])
        
        # 变换源点
        src_pts_homo = np.hstack([src_pts, np.ones((len(src_pts), 1))])
        transformed_pts = (self.H @ src_pts_homo.T).T
        transformed_pts = transformed_pts[:, :2] / transformed_pts[:, 2:3]
        
        # 计算平均误差
        errors = np.linalg.norm(transformed_pts - dst_pts, axis=1)
        avg_error = np.mean(errors)
        
        print(f"平均对齐误差: {avg_error:.2f} 像素")
        return avg_error
    
    def visualize_features(self, pair_name):
        """可视化特征点分布"""
        # 绘制特征点
        line_with_kp = cv2.drawKeypoints(
            self.line_processed, self.kp1, None, 
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        mural_with_kp = cv2.drawKeypoints(
            self.mural_for_matching, self.kp2, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(line_with_kp, cmap='gray')
        plt.title(f'{pair_name} - Line Features ({len(self.kp1)})')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(mural_with_kp, cmap='gray')
        plt.title(f'{pair_name} - Mural Features ({len(self.kp2)})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'results/{pair_name}_features.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_matches(self, pair_name):
        """可视化匹配结果"""
        if len(self.inlier_matches) == 0:
            print("没有内点匹配，尝试可视化所有匹配点")
            if len(self.good_matches) > 0:
                match_img = cv2.drawMatches(
                    self.line_processed, self.kp1, 
                    self.mural_for_matching, self.kp2,
                    self.good_matches, None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                title = f'{pair_name} - All Matches ({len(self.good_matches)})'
            else:
                print("没有匹配点可供可视化")
                return None
        else:
            match_img = cv2.drawMatches(
                self.line_processed, self.kp1, 
                self.mural_for_matching, self.kp2,
                self.inlier_matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            title = f'{pair_name} - Feature Matching (Inliers: {len(self.inlier_matches)}/{len(self.good_matches)})'
        
        # 调整图像大小以适应显示
        height, width = match_img.shape[:2]
        max_height = 800
        if height > max_height:
            scale = max_height / height
            new_width = int(width * scale)
            match_img = cv2.resize(match_img, (new_width, max_height))
        
        plt.figure(figsize=(15, 8))
        plt.imshow(match_img)
        plt.title(title)
        plt.axis('off')
        
        plt.savefig(f'results/{pair_name}_matches.png', dpi=300, bbox_inches='tight')
        plt.close()
        return match_img

def main_stage1():
    """阶段一主函数 - 批量处理6对数据"""
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    # 定义6对数据
    pairs = [
        {"line": "./Data/Line/A-1.jpg", "mural": "./Data/Pic/A-1.png", "name": "A-1"},
        {"line": "./Data/Line/A-2.jpg", "mural": "./Data/Pic/A-2.png", "name": "A-2"},
        {"line": "./Data/Line/A-3.jpg", "mural": "./Data/Pic/A-3.png", "name": "A-3"},
        {"line": "./Data/Line/B-1.jpg", "mural": "./Data/Pic/B-1.png", "name": "B-1"},
        {"line": "./Data/Line/B-2.jpg", "mural": "./Data/Pic/B-2.png", "name": "B-2"},
        {"line": "./Data/Line/B-3.jpg", "mural": "./Data/Pic/B-3.png", "name": "B-3"}
    ]
    
    # 存储所有结果
    all_results = []
    
    for i, pair in enumerate(pairs):
        print(f"\n{'='*60}")
        print(f"处理第 {i+1}/6 对数据: {pair['name']}")
        print(f"{'='*60}")
        
        matcher = TraditionalFeatureMatcher()
        
        # 检查文件是否存在
        line_path_obj = Path(pair["line"])
        mural_path_obj = Path(pair["mural"])
        
        if not line_path_obj.exists():
            print(f"错误：线稿文件不存在 - {pair['line']}")
            continue
        
        if not mural_path_obj.exists():
            print(f"错误：壁画文件不存在 - {pair['mural']}")
            continue
        
        try:
            # 1. 加载图像
            print("1. 加载图像...")
            line_img, mural_img = matcher.load_images(pair["line"], pair["mural"])
            
            # 2. 提取特征
            print("2. 提取特征...")
            kp1, des1, kp2, des2 = matcher.extract_features()
            
            # 可视化特征点分布
            matcher.visualize_features(pair["name"])
            
            # 3. 特征匹配
            print("3. 特征匹配...")
            good_matches = matcher.match_features(ratio_threshold=0.8)
            
            # 4. RANSAC过滤
            print("4. RANSAC过滤...")
            H, inlier_matches = matcher.filter_matches_ransac(reproj_threshold=5.0)
            
            # 5. 计算误差
            print("5. 计算误差...")
            avg_error = matcher.calculate_alignment_error()
            
            # 6. 可视化结果
            print("6. 可视化结果...")
            match_img = matcher.visualize_matches(pair["name"])
            
            # 保存结果
            results = {
                'pair_name': pair["name"],
                'homography_matrix': H,
                'num_matches': len(good_matches),
                'num_inliers': len(inlier_matches),
                'inlier_ratio': matcher.inlier_ratio,
                'avg_error': avg_error,
                'line_features': len(kp1),
                'mural_features': len(kp2)
            }
            
            all_results.append(results)
            
            print(f"\n=== {pair['name']} 结果总结 ===")
            for key, value in results.items():
                if key != 'homography_matrix':  # 不打印矩阵，太长了
                    print(f"{key}: {value}")
            
        except Exception as e:
            print(f"处理 {pair['name']} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 生成汇总报告
    generate_summary_report(all_results)
    
    return all_results

def generate_summary_report(results):
    """生成汇总报告"""
    print(f"\n{'='*80}")
    print("阶段一：传统特征匹配 - 汇总报告")
    print(f"{'='*80}")
    
    # 创建DataFrame便于分析
    df = pd.DataFrame(results)
    
    # 打印汇总统计
    print("\n汇总统计:")
    print(f"处理图像对数量: {len(results)}")
    print(f"平均特征点数量 - 线稿: {df['line_features'].mean():.1f}")
    print(f"平均特征点数量 - 壁画: {df['mural_features'].mean():.1f}")
    print(f"平均匹配点数量: {df['num_matches'].mean():.1f}")
    print(f"平均内点数量: {df['num_inliers'].mean():.1f}")
    print(f"平均内点比例: {df['inlier_ratio'].mean():.3f}")
    print(f"平均对齐误差: {df['avg_error'].mean():.2f} 像素")
    
    # 打印详细结果表格
    print("\n详细结果:")
    print("-" * 80)
    print(f"{'图像对':<10} {'线稿特征点':<12} {'壁画特征点':<12} {'匹配点':<8} {'内点':<8} {'内点比例':<10} {'平均误差':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['pair_name']:<10} {result['line_features']:<12} {result['mural_features']:<12} "
              f"{result['num_matches']:<8} {result['num_inliers']:<8} {result['inlier_ratio']:<10.3f} "
              f"{result['avg_error']:<10.2f}")
    
    print("-" * 80)
    
    # 保存结果到CSV文件
    df.to_csv('results/stage1_results.csv', index=False)
    print(f"\n详细结果已保存到: results/stage1_results.csv")
    
    # 生成可视化图表
    generate_visualization_charts(results)

def generate_visualization_charts(results):
    """生成可视化图表"""
    # 内点比例柱状图
    plt.figure(figsize=(12, 8))
    
    # 子图1: 内点比例
    plt.subplot(2, 2, 1)
    pairs = [r['pair_name'] for r in results]
    inlier_ratios = [r['inlier_ratio'] for r in results]
    plt.bar(pairs, inlier_ratios, color='skyblue')
    plt.title('Inlier Ratio by Image Pair')
    plt.xlabel('Image Pair')
    plt.ylabel('Inlier Ratio')
    plt.xticks(rotation=45)
    
    # 子图2: 特征点数量对比
    plt.subplot(2, 2, 2)
    line_features = [r['line_features'] for r in results]
    mural_features = [r['mural_features'] for r in results]
    x = np.arange(len(pairs))
    width = 0.35
    plt.bar(x - width/2, line_features, width, label='Line Features', color='lightgreen')
    plt.bar(x + width/2, mural_features, width, label='Mural Features', color='lightcoral')
    plt.title('Feature Points Comparison')
    plt.xlabel('Image Pair')
    plt.ylabel('Number of Features')
    plt.xticks(x, pairs, rotation=45)
    plt.legend()
    
    # 子图3: 匹配点数量
    plt.subplot(2, 2, 3)
    matches = [r['num_matches'] for r in results]
    inliers = [r['num_inliers'] for r in results]
    plt.bar(x - width/2, matches, width, label='Total Matches', color='gold')
    plt.bar(x + width/2, inliers, width, label='Inliers', color='orange')
    plt.title('Matching Points Comparison')
    plt.xlabel('Image Pair')
    plt.ylabel('Number of Points')
    plt.xticks(x, pairs, rotation=45)
    plt.legend()
    
    # 子图4: 平均误差
    plt.subplot(2, 2, 4)
    errors = [r['avg_error'] for r in results]
    plt.bar(pairs, errors, color='lightsteelblue')
    plt.title('Average Alignment Error')
    plt.xlabel('Image Pair')
    plt.ylabel('Error (pixels)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/summary_charts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("可视化图表已保存到: results/summary_charts.png")

if __name__ == "__main__":
    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    all_results = main_stage1()