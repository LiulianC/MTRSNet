import cv2
import numpy as np
import os

def create_video_grid(input_paths, output_path, fps=30):
    """
    将多个视频拼接成网格形式
    
    参数:
        input_paths: 输入视频路径列表 (32个)
        output_path: 输出视频路径
        fps: 输出视频帧率
    """
    if len(input_paths) != 32:
        raise ValueError("需要恰好32个输入视频")
    
    caps = [cv2.VideoCapture(path) for path in input_paths]
    
    width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    total_frames = max([int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps])
    
    output_width = width * 8  
    output_height = height * 4  
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    for frame_idx in range(total_frames):
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            frames.append(frame)
        
        grid = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        for i in range(4):  
            for j in range(8):  
                idx = i * 8 + j
                y_start = i * height
                y_end = (i + 1) * height
                x_start = j * width
                x_end = (j + 1) * width
                
                if idx < len(frames):
                    grid[y_start:y_end, x_start:x_end] = frames[idx]
        
        out.write(grid)
        
        if frame_idx % 30 == 0:
            print(f"处理进度: {frame_idx}/{total_frames} 帧")
    
    for cap in caps:
        cap.release()
    out.release()
    
    print(f"视频拼接完成，保存至: {output_path}")
    print(f"输出视频尺寸: {output_width}x{output_height}")

if __name__ == "__main__":
    input_videos = [
        "video/000-in.mp4", "video/001-in.mp4", "video/002-in.mp4", "video/003-in.mp4",
        "video/005-in.mp4", "video/006-in.mp4", "video/008-in.mp4", "video/009-in.mp4",
        "video/011-in.mp4", "video/012-in.mp4", "video/013-in.mp4", "video/014-in.mp4",
        "video/019-in.mp4", "video/020-in.mp4", "video/021-in.mp4", "video/022-in.mp4",
        "video/023-in.mp4", "video/024-in.mp4", "video/025-in.mp4", "video/026-in.mp4",
        "video/027-in.mp4", "video/028-in.mp4", "video/029-in.mp4", "video/030-in.mp4",
        "video/031-in.mp4", "video/032-in.mp4", "video/033-in.mp4", "video/035-in.mp4",
        "video/036-in.mp4", "video/037-in.mp4", "video/038-in.mp4", "video/040-in.mp4",
    ]

    for path in input_videos:
        if not os.path.exists(path):
            print(f"警告: 找不到视频文件: {path}")
    
    create_video_grid(input_videos, "output_grid.mp4", fps=30)