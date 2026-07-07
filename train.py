import os
import sys
import time
import torch
from pathlib import Path
from ultralytics import YOLO

# ==========================================
# 0. 工业级显存碎片化优化 (适配 RTX 5090 D Nightly 版)
# ==========================================
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def check_environment(data_yaml: Path):
    """训练前的前置环境与数据安全预检"""
    print("="*50)
    print("🔍 正在执行工业级训练前预检...")
    
    if not data_yaml.exists():
        print(f"❌ 致命错误: 数据集配置文件不存在 {data_yaml}")
        sys.exit(1)
    print("✅ 数据集配置文件存在。")
    
    if not torch.cuda.is_available():
        print("❌ 致命错误: 未检测到可用 GPU，无法启动满血训练。")
        sys.exit(1)
        
    gpu_name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    print(f"✅ GPU 就绪: {gpu_name} (算力 CC: {cc[0]}.{cc[1]})")
    
    # 清理显存碎片
    torch.cuda.empty_cache()
    print("✅ 显存池已优化。")
    print("="*50)

def main():
    # ==========================================
    # 1. 模型与路径定义
    # ==========================================
    data_yaml = Path('/home/ai/data_disk1/yolo/dataset/dataset.yaml')
    project_dir = Path('/home/ai/data_disk1/yolo/runs/train')
    exp_name = 'finish_exp1'
    
    # 执行预检
    check_environment(data_yaml)
    
    start_time = time.time()
    
    try:
        # 采用 YOLO11m 版本，完美榨干 RTX 5090 D 算力
        print(f"🚀 启动 RTX 5090 D 满血训练 (实验名: {exp_name})...")
        model = YOLO('yolo11m.pt') 
        
        # ==========================================
        # 2. 启动训练 (融合前沿调参策略)
        # ==========================================
        results = model.train(
            data=str(data_yaml),
            epochs=150,          # 训练轮数
            imgsz=640,           # 标准工业输入分辨率
            batch=-1,            # 🌟 AutoBatch: 自动探测 5090D 显存，动态分配最大不溢出的 Batch Size
            device=0,            # 强制使用 GPU 0
            workers=16,          # 配合 9950X 32线程，极速数据加载
            project=str(project_dir),
            name=exp_name,       # 实验名称
            patience=30,         # 30轮验证精度不提升则自动早停，节省算力
            cos_lr=True,         # 🌟 启用余弦退火学习率，平滑收敛
            
            # 🌟 高级技术：末端关闭 Mosaic
            # 前 140 轮开启 Mosaic 增加泛化，最后 10 轮关闭进行真实分布微调
            close_mosaic=10,        
            
            save=True,           # 保存权重
            plots=True,          # 自动生成 results.png 等可视化图表
            verbose=True
        )

        # ==========================================
        # 3. 训练后自动验证 (闭环测试)
        # ==========================================
        print("\n" + "="*50)
        print("📊 训练完成，正在加载最佳模型进行验证...")
        best_model_path = project_dir / exp_name / 'weights' / 'best.pt'
        
        if best_model_path.exists():
            best_model = YOLO(str(best_model_path))
            metrics = best_model.val(split='val')
            
            print("\n" + "="*50)
            print("✅ 最终验证结果:")
            print(f"   - 精确率:    {metrics.box.mp:.4f}")
            print(f"   - 召回率:    {metrics.box.mr:.4f}")
            print(f"   - mAP50:      {metrics.box.map50:.4f}")
            print(f"   - mAP50-95:   {metrics.box.map:.4f}")
            print("="*50)
            
            # ==========================================
            # 4. 自动导出工业部署模型 (ONNX)
            # ==========================================
            print("📦 正在导出 ONNX 工业部署模型...")
            try:
                onnx_path = best_model.export(format='onnx', simplify=True)
                print(f"✅ ONNX 导出成功: {onnx_path}")
            except Exception as export_err:
                print(f"⚠️ ONNX 导出失败 (不影响训练结果): {export_err}")

        else:
            print("⚠️ 未找到最佳模型权重，请检查训练是否异常中断。")

    except torch.cuda.OutOfMemoryError:
        print("\n" + "="*50)
        print("❌ 致命错误: GPU 显存溢出 (OOM)")
        print("建议: 虽然启用了 AutoBatch，但仍遇极端批次溢出，可手动在代码中强制设置 batch=8 或 4。")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n" + "="*50)
        print("⚠️ 用户手动中断训练")
        last_pt = project_dir / exp_name / 'weights' / 'last.pt'
        if last_pt.exists():
            print(f"💡 提示: 您可以使用以下命令从断点恢复训练:")
            print(f"   yolo resume model={last_pt}")
        sys.exit(0)
        
    except Exception as e:
        print("\n" + "="*50)
        print(f"❌ 发生未知异常: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        end_time = time.time()
        duration = (end_time - start_time) / 60
        print(f"\n⏱️ 本次任务总耗时: {duration:.2f} 分钟")

if __name__ == '__main__':
    main()