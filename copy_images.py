import os
import shutil
# 只修改num就可以了

num = 4




# 原始三个文件夹
source_dirs = [
    fr"C:\Users\57746\Desktop\newpipeline\images\batch{num}\0", 
    fr"C:\Users\57746\Desktop\newpipeline\images\batch{num}\1", 
    fr"C:\Users\57746\Desktop\newpipeline\images\batch{num}\2"
]

# 目标文件夹
top_dir = fr"C:\Users\57746\Desktop\newpipeline\images\batch{num}\top_images"
bottom_dir = fr"C:\Users\57746\Desktop\newpipeline\images\batch{num}\bottom_images"

# 创建目标文件夹（如果不存在）
os.makedirs(top_dir, exist_ok=True)
os.makedirs(bottom_dir, exist_ok=True)

for index, src in enumerate(source_dirs):
    # 按 source_dirs 的顺序使用 0、1、2 作为前缀
    folder_name = str(index)

    # 只处理文件，避免把子目录也算进去
    images = sorted(
        name for name in os.listdir(src)
        if os.path.isfile(os.path.join(src, name))
    )

    if len(images) < 32:
        print(f"警告: {src} 只有 {len(images)} 张图片，少于 32 张，后 16 张将为空")
    
    # 前16张 -> 复制到 top_images
    for img in images[:16]:
        src_path = os.path.join(src, img)
        dst_path = os.path.join(top_dir, f"{folder_name}_{img}")  # 加前缀避免重名
        shutil.copy2(src_path, dst_path)

    # 后16张 -> 复制到 bottom_images
    for img in images[16:]:
        src_path = os.path.join(src, img)
        dst_path = os.path.join(bottom_dir, f"{folder_name}_{img}")
        shutil.copy2(src_path, dst_path)

print("处理完成！")