import sys
import cv2
import numpy as np
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QSlider, 
                               QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
                               QDoubleSpinBox, QGroupBox, QMessageBox)
from ultralytics import YOLO
from PIL import Image as PILImage

def format_time(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

# ==========================================
# 1. 工业级防重复状态机
# ==========================================
class ActionStateMachine:
    def __init__(self, confirm_frames=3, cooldown_time=1.5, max_cycle_time=60.0):
        self.state = 'IDLE'
        self.frame_counter = 0
        self.confirm_frames = confirm_frames
        
        self.cooldown_time = cooldown_time     
        self.max_cycle_time = max_cycle_time   
        
        self.count = 0
        self.break_count = 0                   
        self.last_action_start_time = None     
        
        self.all_intervals = []               
        
        self.max_processed_frame = -1     
        self.is_frozen = False            
        self.last_safe_frame = 0          
        self.cooldown_start_time = 0.0         

    def reset_all(self):
        old_cooldown = self.cooldown_time
        old_max_cycle = self.max_cycle_time
        self.__init__()
        self.cooldown_time = old_cooldown
        self.max_cycle_time = old_max_cycle

    def update(self, detected, current_time_sec, current_frame):
        if current_frame < self.max_processed_frame:
            self.is_frozen = True
            return
        else:
            self.max_processed_frame = current_frame
            self.is_frozen = False
            self.last_safe_frame = current_frame

        if self.state == 'IDLE':
            if detected:
                self.frame_counter += 1
                if self.frame_counter >= self.confirm_frames:
                    if self.last_action_start_time is not None:
                        interval = current_time_sec - self.last_action_start_time
                        if interval > 0:
                            is_abnormal = interval > self.max_cycle_time
                            if is_abnormal:
                                self.break_count += 1
                            self.all_intervals.append((interval, self.count + 1, self.last_action_start_time, current_time_sec, is_abnormal))
                    
                    self.count += 1
                    self.state = 'LOCKED'
                    self.last_action_start_time = current_time_sec
                    self.frame_counter = 0
            else:
                self.frame_counter = 0

        elif self.state == 'LOCKED':
            if not detected:
                self.state = 'COOLDOWN'
                self.cooldown_start_time = current_time_sec

        elif self.state == 'COOLDOWN':
            if detected:
                self.state = 'LOCKED'
            else:
                if (current_time_sec - self.cooldown_start_time) >= self.cooldown_time:
                    self.state = 'IDLE'

    def get_stats(self):
        normal_intervals = [item[0] for item in self.all_intervals if not item[4]]
        avg_time = np.mean(normal_intervals) if normal_intervals else 0.0
        top_n_max = sorted(self.all_intervals, key=lambda x: x[0], reverse=True)[:10]
        top_n_min = sorted([item for item in self.all_intervals if not item[4]], key=lambda x: x[0])[:10]
        return self.count, self.break_count, avg_time, top_n_max, top_n_min

# ==========================================
# 2. 🚀 极速预处理后台线程 (QThread + 批处理)
# ==========================================
class PreprocessThread(QThread):
    progress = Signal(int, str)
    finished = Signal(object)

    def __init__(self, video_path, model_path, state_machine):
        super().__init__()
        self.video_path = video_path
        self.model_path = model_path
        self.state_machine = state_machine
        self.batch_size = 32

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        try:
            model = YOLO(self.model_path)
        except Exception:
            cap.release()
            return

        batch_frames = []
        batch_frame_ids = []
        current_frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            batch_frames.append(frame)
            batch_frame_ids.append(current_frame_idx)
            current_frame_idx += 1

            if len(batch_frames) == self.batch_size or not ret:
                results = model.predict(batch_frames, conf=0.5, verbose=False, batch=self.batch_size)
                
                for i, res in enumerate(results):
                    detected = res.boxes.cls.numel() > 0
                    time_sec = batch_frame_ids[i] / fps
                    self.state_machine.update(detected, time_sec, batch_frame_ids[i])
                
                prog = int((current_frame_idx / total_frames) * 100)
                time_str = f"{format_time(current_frame_idx / fps)} / {format_time(total_frames / fps)}"
                self.progress.emit(prog, time_str)
                
                batch_frames.clear()
                batch_frame_ids.clear()

        cap.release()
        self.finished.emit(self.state_machine)

# ==========================================
# 3. 主窗口 (异步极速版)
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🛠️ 工业视觉效能透视终端 V3.0 (异步极速版)")
        self.resize(1600, 950)
        self.setStyleSheet("background-color: #1e1e1e; color: #ffffff; font-family: 'Segoe UI', Arial;")
        
        self.model = None
        self.cap = None
        self.fps = 30.0
        self.total_frames = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.state_machine = ActionStateMachine(confirm_frames=3, cooldown_time=1.5, max_cycle_time=60.0)
        self.is_playing = False
        self.is_preprocessing = False
        self.preprocess_thread = None

        self._init_ui()
        self.load_model()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        ctrl_layout = QHBoxLayout()
        self.btn_load = QPushButton("📁 导入视频")
        self.btn_load.setStyleSheet("background-color: #007acc; padding: 12px 24px; font-size: 16px; font-weight: bold; border-radius: 6px;")
        self.btn_load.clicked.connect(self.load_video)
        
        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.setStyleSheet("background-color: #28a745; padding: 12px 24px; font-size: 16px; font-weight: bold; border-radius: 6px;")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setEnabled(False)

        self.btn_export = QPushButton("📤 导出Excel报告")
        self.btn_export.setStyleSheet("background-color: #ff8c00; padding: 12px 24px; font-size: 16px; font-weight: bold; border-radius: 6px;")
        self.btn_export.clicked.connect(self.export_report)
        self.btn_export.setEnabled(False)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setStyleSheet("height: 25px;")
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self.set_video_pos)

        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setStyleSheet("font-size: 16px; color: #aaa; min-width: 120px;")

        ctrl_layout.addWidget(self.btn_load)
        ctrl_layout.addWidget(self.btn_play)
        ctrl_layout.addWidget(self.btn_export)
        ctrl_layout.addWidget(self.slider)
        ctrl_layout.addWidget(self.lbl_time)
        main_layout.addLayout(ctrl_layout)

        self.lbl_safe_point = QLabel("💡 请导入视频开始分析")
        self.lbl_safe_point.setStyleSheet("font-size: 14px; color: #aaaaaa; background-color: #333; padding: 5px; border-radius: 4px;")
        main_layout.addWidget(self.lbl_safe_point)

        video_layout = QHBoxLayout()
        self.label_original = QLabel("等待导入视频...")
        self.label_ai = QLabel("AI 透视镜")
        for label in [self.label_original, self.label_ai]:
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("background-color: #000000; border: 2px solid #333333; font-size: 20px; color: #666;")
            label.setMinimumSize(700, 500)
        video_layout.addWidget(self.label_original)
        video_layout.addWidget(self.label_ai)
        main_layout.addLayout(video_layout, stretch=3)

        stats_layout = QHBoxLayout()
        stats_left = QVBoxLayout()
        
        self.lbl_count = QLabel("总计数: 0")
        self.lbl_count.setStyleSheet("font-size: 28px; font-weight: bold; color: #00ffcc; padding: 5px;")
        self.lbl_break = QLabel("异常行为: 0 次")
        self.lbl_break.setStyleSheet("font-size: 20px; font-weight: bold; color: #ff8800; padding: 5px;")
        self.lbl_avg = QLabel("平均周期: 0.00 秒")
        self.lbl_avg.setStyleSheet("font-size: 24px; font-weight: bold; color: #00ffcc; padding: 5px;")
        
        stats_left.addWidget(self.lbl_count)
        stats_left.addWidget(self.lbl_break)
        stats_left.addWidget(self.lbl_avg)

        threshold_group = QGroupBox("⚙️ 核心阈值实时调控")
        threshold_group.setStyleSheet("QGroupBox { border: 1px solid #555; margin-top: 10px; padding: 10px; font-weight: bold; color: #ddd; }")
        threshold_layout = QVBoxLayout()
        
        cooldown_layout = QHBoxLayout()
        lbl_cool = QLabel("防抖冷却 (秒):")
        lbl_cool.setStyleSheet("font-size: 14px;")
        self.spin_cooldown = QDoubleSpinBox()
        self.spin_cooldown.setRange(0.1, 3600.0)
        self.spin_cooldown.setSingleStep(0.1)
        self.spin_cooldown.setValue(1.5)
        self.spin_cooldown.setStyleSheet("font-size: 16px; padding: 5px;")
        self.spin_cooldown.valueChanged.connect(self.update_thresholds)
        cooldown_layout.addWidget(lbl_cool)
        cooldown_layout.addWidget(self.spin_cooldown)
        threshold_layout.addLayout(cooldown_layout)

        break_layout = QHBoxLayout()
        lbl_break_t = QLabel("异常熔断 (秒):")
        lbl_break_t.setStyleSheet("font-size: 14px;")
        self.spin_break = QDoubleSpinBox()
        self.spin_break.setRange(0.1, 3600.0)
        self.spin_break.setSingleStep(0.1)
        self.spin_break.setValue(60.0)
        self.spin_break.setStyleSheet("font-size: 16px; padding: 5px;")
        self.spin_break.valueChanged.connect(self.update_thresholds)
        break_layout.addWidget(lbl_break_t)
        break_layout.addWidget(self.spin_break)
        threshold_layout.addLayout(break_layout)
        
        threshold_group.setLayout(threshold_layout)
        stats_left.addWidget(threshold_group)
        stats_left.addStretch()
        stats_layout.addLayout(stats_left, stretch=1)

        tables_layout = QHBoxLayout()
        max_layout = QVBoxLayout()
        max_lbl = QLabel("🚨 耗时最长 Top 10 (含异常)")
        max_lbl.setStyleSheet("font-size: 18px; font-weight: bold; color: #ff4444; padding: 5px;")
        max_layout.addWidget(max_lbl)
        self.table_max = QTableWidget(10, 4)
        self.table_max.setHorizontalHeaderLabels(["排名", "动作序号", "耗时(秒)", "视频进度"])
        self.table_max.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_max.verticalHeader().setVisible(False)
        self.table_max.setStyleSheet("font-size: 14px; background-color: #2d2d2d;")
        self.table_max.cellClicked.connect(self.jump_to_action)
        max_layout.addWidget(self.table_max)
        tables_layout.addLayout(max_layout, stretch=1)

        min_layout = QVBoxLayout()
        min_lbl = QLabel("🏆 效率最高 Top 10 (正常节拍)")
        min_lbl.setStyleSheet("font-size: 18px; font-weight: bold; color: #00ff00; padding: 5px;")
        min_layout.addWidget(min_lbl)
        self.table_min = QTableWidget(10, 4)
        self.table_min.setHorizontalHeaderLabels(["排名", "动作序号", "耗时(秒)", "视频进度"])
        self.table_min.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_min.verticalHeader().setVisible(False)
        self.table_min.setStyleSheet("font-size: 14px; background-color: #2d2d2d;")
        self.table_min.cellClicked.connect(self.jump_to_action)
        min_layout.addWidget(self.table_min)
        tables_layout.addLayout(min_layout, stretch=1)

        stats_layout.addLayout(tables_layout, stretch=2.5)
        main_layout.addLayout(stats_layout, stretch=1)

    def export_report(self):
        if not self.state_machine.all_intervals:
            QMessageBox.information(self, "提示", "当前无数据可导出！")
            return
            
        export_dir = "/home/ai/data_disk1/yolo/"
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(export_dir, f"效能分析报告_{timestamp}.xlsx")
        
        count, break_count, avg_time, top_n_max, top_n_min = self.state_machine.get_stats()
        
        df_summary = pd.DataFrame({
            "指标": ["总计数", "异常行为次数", "平均周期(秒)", "正常节拍数"],
            "数值": [count, break_count, f"{avg_time:.2f}", count - break_count]
        })
        
        detail_data = []
        for interval, action_idx, start_t, end_t, is_abnormal in self.state_machine.all_intervals:
            detail_data.append({
                "动作序号": action_idx, "耗时(秒)": round(interval, 2),
                "开始时间": format_time(start_t), "结束时间": format_time(end_t),
                "是否异常": "是" if is_abnormal else "否"
            })
        df_detail = pd.DataFrame(detail_data)
        
        try:
            img_path = os.path.join(export_dir, f"temp_chart_{timestamp}.png")
            self.generate_charts(df_detail, img_path)
            abs_img_path = os.path.abspath(img_path)
            
            with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
                df_summary.to_excel(writer, sheet_name='汇总数据', index=False)
                df_detail.to_excel(writer, sheet_name='动作明细', index=False)
                
                workbook = writer.book
                worksheet = workbook.add_worksheet('可视化图表')
                worksheet.insert_image('A1', abs_img_path, {'x_scale': 0.8, 'y_scale': 0.8})
            
            if os.path.exists(img_path):
                os.remove(img_path)
                
            QMessageBox.information(self, "导出成功", f"报告已成功导出至:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"发生错误: {e}")

    def generate_charts(self, df_detail, img_path):
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), dpi=100)
        fig.suptitle('Industrial Efficiency Analysis Report', fontsize=20, color='#00ffcc', weight='bold')
        
        ax1 = axes[0]
        normal_df = df_detail[df_detail['是否异常'] == '否']
        abnormal_df = df_detail[df_detail['是否异常'] == '是']
        
        if not normal_df.empty:
            ax1.plot(normal_df['动作序号'], normal_df['耗时(秒)'], color='#00ffcc', marker='o', markersize=4, label='Normal Cycle', linestyle='-')
        if not abnormal_df.empty:
            ax1.scatter(abnormal_df['动作序号'], abnormal_df['耗时(秒)'], color='#ff4444', s=80, marker='v', label='Abnormal Timeout', zorder=5)
            
        ax1.set_title('Action Duration Trend (Stability & Fatigue)', fontsize=14, color='white')
        ax1.set_xlabel('Action Index', fontsize=12, color='white')
        ax1.set_ylabel('Duration (s)', fontsize=12, color='white')
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        ax2 = axes[1]
        if not df_detail.empty:
            max_time = int(df_detail['耗时(秒)'].max()) + 2
            bins = range(0, max_time, max(1, int(max_time/15)))
            ax2.hist(df_detail['耗时(秒)'], bins=bins, color='#007acc', edgecolor='white', alpha=0.8)
            
            avg = df_detail[df_detail['是否异常'] == '否']['耗时(秒)'].mean()
            if not np.isnan(avg):
                ax2.axvline(avg, color='#00ff00', linestyle='--', linewidth=2, label=f'Avg Normal: {avg:.2f}s')
                
            ax2.set_title('Duration Frequency Distribution (Standard Time)', fontsize=14, color='white')
            ax2.set_xlabel('Duration Range (s)', fontsize=12, color='white')
            ax2.set_ylabel('Frequency', fontsize=12, color='white')
            ax2.legend(loc='upper right')
            ax2.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        temp_path = img_path + "_raw.png"
        plt.savefig(temp_path)
        plt.close(fig) 
        
        pil_img = PILImage.open(temp_path)
        rgb_im = pil_img.convert('RGB')
        rgb_im.save(img_path, 'PNG')
            
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def update_thresholds(self):
        self.state_machine.cooldown_time = self.spin_cooldown.value()
        self.state_machine.max_cycle_time = self.spin_break.value()

    def load_model(self):
        model_path = "/home/ai/data_disk1/yolo/runs/train/finish_exp1/weights/best.pt"
        if not Path(model_path).exists():
            self.label_ai.setText("❌ 未找到模型 best.pt\n请先完成模型训练！")
            return False
        try:
            self.model = YOLO(model_path)
            return True
        except Exception as e:
            self.label_ai.setText(f"模型加载失败: {e}")
            return False

    def load_video(self):
        if not self.model: return
            
        video_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if not video_path: return
            
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened(): return
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.slider.setRange(0, self.total_frames - 1)
        self.slider.setValue(0)
        self.slider.setEnabled(False) 
        self.btn_play.setEnabled(False)
        self.btn_export.setEnabled(False)
        
        self.state_machine.reset_all()
        self.update_stats_ui()
        
        self.is_preprocessing = True
        self.label_original.setText("🚀 GPU 批处理加速中...")
        self.label_ai.setText("🚀 GPU 批处理加速中...")
        self.lbl_safe_point.setText("🚀 正在调用 RTX 5090 D 极速预处理 (Batch=32)，请稍候...")
        self.lbl_safe_point.setStyleSheet("font-size: 14px; color: #00aaff; background-color: #333; padding: 5px; border-radius: 4px;")
        
        self.preprocess_thread = PreprocessThread(video_path, "/home/ai/data_disk1/yolo/runs/train/finish_exp1/weights/best.pt", self.state_machine)
        self.preprocess_thread.progress.connect(self.on_preprocess_progress)
        self.preprocess_thread.finished.connect(self.on_preprocess_finished)
        self.preprocess_thread.start()

    def on_preprocess_progress(self, prog, time_str):
        self.slider.setValue(prog)
        self.lbl_time.setText(f"分析中... {time_str}")

    def on_preprocess_finished(self, state_machine):
        self.state_machine = state_machine
        self.is_preprocessing = False
        
        self.slider.setEnabled(True)
        self.slider.setValue(0)
        self.btn_play.setEnabled(True)
        self.btn_export.setEnabled(True)
        
        self.lbl_time.setText(f"00:00 / {format_time(self.total_frames / self.fps)}")
        
        self.label_original.setText("✅ 预处理完成\n\n请点击 ▶ 播放 查看追踪回放\n或点击底部榜单跳转审查")
        self.label_ai.setText("✅ 预处理完成\n\n当前处于安全审查模式 (不计数)\n请点击 ▶ 播放")
        
        self.lbl_safe_point.setText("✅ 预处理已完成！数据已锁定。当前播放/拖拽仅用于审查，绝不重复计数。")
        self.lbl_safe_point.setStyleSheet("font-size: 14px; color: #00ff00; background-color: #333; padding: 5px; border-radius: 4px;")
        self.update_stats_ui()

    def read_and_show_specific_frame(self, frame_idx):
        if not self.cap: return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            ai_frame = frame.copy()
            detected = False
            if self.model:
                results = self.model.predict(frame, conf=0.5, verbose=False)
                if results[0].boxes.cls.numel() > 0:
                    detected = True
                    ai_frame = results[0].plot()
            
            count = self.state_machine.count
            cv2.putText(ai_frame, f"Count: {count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            if self.state_machine.is_frozen:
                cv2.putText(ai_frame, "State: REVIEW MODE (No Counting)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
            else:
                cv2.putText(ai_frame, f"State: {self.state_machine.state}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            
            self.show_frame(frame, ai_frame)

    def toggle_play(self):
        if self.is_preprocessing: return
        if not self.cap: return
        
        self.is_playing = not self.is_playing
        self.btn_play.setText("⏸ 暂停" if self.is_playing else "▶ 播放")
        if self.is_playing:
            self.timer.start(int(1000 / self.fps))
        else:
            self.timer.stop()

    def update_frame(self):
        if not self.cap or not self.is_playing: return
        
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.is_playing = False
            self.btn_play.setText("▶ 播放 (已完成)")
            return

        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.slider.setValue(current_frame)
        current_time_sec = current_frame / self.fps
        
        self.lbl_time.setText(f"{format_time(current_time_sec)} / {format_time(self.total_frames / self.fps)}")

        ai_frame = frame.copy()
        detected = False
        if self.model:
            results = self.model.predict(frame, conf=0.5, verbose=False)
            if results[0].boxes.cls.numel() > 0:
                detected = True
                ai_frame = results[0].plot()

        self.state_machine.update(detected, current_time_sec, current_frame)
        count = self.state_machine.count
        
        cv2.putText(ai_frame, f"Count: {count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        if self.state_machine.is_frozen:
            cv2.putText(ai_frame, "State: REVIEW MODE (No Counting)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
            self.lbl_safe_point.setText("🛡️ 安全审查中 ")
            self.lbl_safe_point.setStyleSheet("font-size: 14px; color: #ffaa00; background-color: #333; padding: 5px; border-radius: 4px;")
        else:
            cv2.putText(ai_frame, f"State: {self.state_machine.state}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        self.show_frame(frame, ai_frame)
        self.update_stats_ui()

    def set_video_pos(self, position):
        if not self.cap: return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        self.lbl_time.setText(f"{format_time(position / self.fps)} / {format_time(self.total_frames / self.fps)}")
        if not self.is_playing:
            self.read_and_show_specific_frame(position)

    def jump_to_action(self, row, col):
        if not self.cap: return
        table = self.sender()
        if not table: return
        
        time_item = table.item(row, 3) 
        if time_item:
            start_time_str = time_item.text().split(" - ")[0]
            mins, secs = map(int, start_time_str.split(":"))
            target_sec = mins * 60 + secs
            
            target_frame = int(target_sec * self.fps)
            self.slider.setValue(target_frame)
            self.set_video_pos(target_frame)

    def show_frame(self, original_img, ai_img):
        rgb_original = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        rgb_ai = cv2.cvtColor(ai_img, cv2.COLOR_BGR2RGB)
        
        h, w, ch = rgb_original.shape
        bytes_per_line = ch * w
        qimg_original = QImage(rgb_original.data, w, h, bytes_per_line, QImage.Format_RGB888)
        qimg_ai = QImage(rgb_ai.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        self.label_original.setPixmap(QPixmap.fromImage(qimg_original).scaled(
            self.label_original.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.label_ai.setPixmap(QPixmap.fromImage(qimg_ai).scaled(
            self.label_ai.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_stats_ui(self):
        count, break_count, avg_time, top_n_max, top_n_min = self.state_machine.get_stats()
        self.lbl_count.setText(f"总计数: {count}")
        self.lbl_break.setText(f"异常行为: {break_count} 次")
        self.lbl_avg.setText(f"平均周期: {avg_time:.2f} 秒")
        
        self.table_max.setRowCount(0)
        for i, (interval, action_idx, start_t, end_t, is_abnormal) in enumerate(top_n_max):
            self.table_max.insertRow(i)
            self.table_max.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            
            item_act = QTableWidgetItem(f"{action_idx} (异常)" if is_abnormal else str(action_idx))
            if is_abnormal:
                item_act.setForeground(QColor("orange"))
            self.table_max.setItem(i, 1, item_act)
            
            item_time = QTableWidgetItem(f"{interval:.2f}")
            if is_abnormal:
                item_time.setForeground(QColor("orange"))
            self.table_max.setItem(i, 2, item_time)
            
            time_range = f"{format_time(start_t)} - {format_time(end_t)}"
            self.table_max.setItem(i, 3, QTableWidgetItem(time_range))

        self.table_min.setRowCount(0)
        for i, (interval, action_idx, start_t, end_t, is_abnormal) in enumerate(top_n_min):
            self.table_min.insertRow(i)
            self.table_min.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.table_min.setItem(i, 1, QTableWidgetItem(str(action_idx)))
            self.table_min.setItem(i, 2, QTableWidgetItem(f"{interval:.2f}"))
            time_range = f"{format_time(start_t)} - {format_time(end_t)}"
            self.table_min.setItem(i, 3, QTableWidgetItem(time_range))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())