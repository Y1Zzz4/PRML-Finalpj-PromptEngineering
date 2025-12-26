import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

class ARCVisualizer:
    def __init__(self):
        # ARC 标准颜色十六进制
        self.hex_colors = [
            '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
            '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
        ]
        # 用于 matplotlib 的颜色映射
        self.cmap = mcolors.ListedColormap(self.hex_colors)
        self.norm = mcolors.BoundaryNorm(np.arange(-0.5, 10.5, 1), self.cmap.N)

    def clear_visuals_dir(self, visuals_dir):
        """运行前清空旧的可视化文件，确保数据一致性"""
        if os.path.exists(visuals_dir):
            shutil.rmtree(visuals_dir)
        os.makedirs(visuals_dir)
        print(f"已清空旧的可视化目录: {visuals_dir}")

    def save_png(self, grid, filename, title):
        """生成 PNG 图片"""
        if not grid: return
        data = np.array(grid)
        h, w = data.shape
        
        # 动态计算尺寸，保证网格是方的
        fig, ax = plt.subplots(figsize=(w*0.5, h*0.5))
        ax.imshow(data, cmap=self.cmap, norm=self.norm)
        
        # 绘制白色网格线
        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
        
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        plt.title(title)
        plt.savefig(filename, bbox_inches='tight', dpi=100)
        plt.close()

    def generate_tikz(self, grid, scale=0.4):
        """生成 LaTeX TikZ 代码"""
        if not grid: return "% Invalid Grid"
        rows, cols = len(grid), len(grid[0])
        tikz_lines = [f"\\begin{{tikzpicture}}[scale={scale}]"]
        for r in range(rows):
            for c in range(cols):
                tikz_lines.append(f"  \\fill[arc{grid[r][c]}, draw=white, line width=0.2pt] ({c}, {-r}) rectangle ++(1, 1);")
        tikz_lines.append("\\end{tikzpicture}")
        return "\n".join(tikz_lines)

    def save_task_visuals(self, task_dir, task_id, grid, type_label):
        """同时保存 .tex 和 .png"""
        if not grid: return

        # 1. 定义子路径
        png_folder = os.path.join(task_dir, "png")
        tex_folder = os.path.join(task_dir, "tex")

        # 2. 自动创建目录
        os.makedirs(png_folder, exist_ok=True)
        os.makedirs(tex_folder, exist_ok=True)

        # 3. 保存 PNG
        png_filename = os.path.join(png_folder, f"{type_label.lower()}.png")
        self.save_png(grid, png_filename, f"Task {task_id}: {type_label}")

        # 4. 保存 TEX
        tex_filename = os.path.join(tex_folder, f"{type_label.lower()}.tex")
        with open(tex_filename, 'w', encoding='utf-8') as f:
            f.write(self.generate_tikz(grid))