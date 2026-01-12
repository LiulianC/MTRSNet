import cv2
import numpy as np

def create_transition_video(video1_path, video2_path, output_path, 
                           start_time=0, move_duration=5, stay_duration=3, stay_duration2=3,
                           glow_intensity=0.3):
    """
    创建带有来回移动蒙版效果的视频过渡
    
    参数:
        video1_path: 上方视频路径（原始视频）
        video2_path: 下方视频路径（处理后的视频）
        output_path: 输出视频路径
        start_time: 效果开始的时间（秒）
        move_duration: 蒙版移动的持续时间（秒）
        stay_duration: 完全显示下方视频的持续时间（秒）
        glow_intensity: 发光效果强度（0-1）
    """
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    if not cap1.isOpened() or not cap2.isOpened():
        print("错误：无法打开视频文件")
        return
    
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start_frame = int(start_time * fps)
    
    move_frames = int(move_duration * fps)
    stay_frames = int(stay_duration * fps)
    stay_frames2 = int(stay_duration2 * fps)
    
    cap1.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    def create_glow_mask(width, height, position):
        mask = np.zeros((height, width), dtype=np.uint8)
        start = max(0, position - 15)
        end = min(width, position + 15)
        cv2.rectangle(mask, (start, 0), (end, height), 255, -1)
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    print("第二阶段：完全显示处理后的视频")
    for frame_idx in range(stay_frames):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        combined_frame = frame2.copy()
        
        cv2.line(combined_frame, (width-1, 0), (width-1, height), 
                (0, 255, 255), 3)
        
        cv2.putText(combined_frame, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(combined_frame, "Processed", (width - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        current_time = start_time + (frame_idx) / fps
        time_text = f"Time: {current_time:.2f}s"
        cv2.putText(combined_frame, time_text, (width // 2 - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        out.write(combined_frame)
        
        if frame_idx % 30 == 0:
            print(f"处理进度: {frame_idx}/{stay_frames} 帧")

    print("第一阶段：蒙版从左向右移动")
    for frame_idx in range(move_frames):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        split_position = int(width * frame_idx / move_frames)
        
        combined_frame = frame2.copy()
        
        combined_frame[:, :split_position] = frame1[:, :split_position]
        
        glow_mask = create_glow_mask(width, height, split_position)
        
        combined_frame = cv2.addWeighted(combined_frame, 1.0, glow_mask, glow_intensity, 0)
        
        cv2.line(combined_frame, (split_position, 0), (split_position, height), 
                (0, 255, 255), 3)
        
        cv2.putText(combined_frame, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(combined_frame, "Processed", (width - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        current_time = start_time + (stay_frames+frame_idx) / fps
        time_text = f"Time: {current_time:.2f}s"
        cv2.putText(combined_frame, time_text, (width // 2 - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        out.write(combined_frame)
        
        if frame_idx % 30 == 0:
            print(f"处理进度: {frame_idx}/{move_frames} 帧")

    
    print("第三阶段：蒙版从右向左移动")
    for frame_idx in range(move_frames):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        split_position = width - int(width * frame_idx / move_frames)
        
        combined_frame = frame2.copy()
        
        combined_frame[:, :split_position] = frame1[:, :split_position]
        
        glow_mask = create_glow_mask(width, height, split_position)
        
        combined_frame = cv2.addWeighted(combined_frame, 1.0, glow_mask, glow_intensity, 0)
        
        cv2.line(combined_frame, (split_position, 0), (split_position, height), 
                (0, 255, 255), 3)
        
        cv2.putText(combined_frame, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(combined_frame, "Processed", (width - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        current_time = start_time + (move_frames + stay_frames + frame_idx) / fps
        time_text = f"Time: {current_time:.2f}s"
        cv2.putText(combined_frame, time_text, (width // 2 - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        out.write(combined_frame)
        
        if frame_idx % 30 == 0:
            print(f"处理进度: {frame_idx}/{move_frames} 帧")
    
    print("第四阶段：完全显示处理后的视频")
    for frame_idx in range(stay_frames2):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        combined_frame = frame2.copy()
        
        cv2.line(combined_frame, (width-1, 0), (width-1, height), 
                (0, 255, 255), 3)
        
        cv2.putText(combined_frame, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(combined_frame, "Processed", (width - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        current_time = start_time + (move_frames*2 + stay_frames + frame_idx) / fps
        time_text = f"Time: {current_time:.2f}s"
        cv2.putText(combined_frame, time_text, (width // 2 - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        out.write(combined_frame)
        
        if frame_idx % 30 == 0:
            print(f"处理进度: {frame_idx}/{stay_frames} 帧")

    cap1.release()
    cap2.release()
    out.release()
    print(f"视频处理完成！已保存至: {output_path}")
    total_effect_frames = move_frames + stay_frames + move_frames
    print(f"处理了 {total_effect_frames} 帧，时长: {total_effect_frames/fps:.2f} 秒")

if __name__ == "__main__":
    original_video = "output.mp4"
    processed_video = "input.mp4"
    output_video = "transition_effect.mp4"
    
    create_transition_video(
        original_video, 
        processed_video, 
        output_video,
        start_time=0,         
        stay_duration=3,       
        move_duration=7,      
        stay_duration2=3,
        glow_intensity=0.05,
    )