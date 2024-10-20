import torch
import time

def occupy_gpu_memory(device_id=0):
    # 设置设备，默认为第一个 GPU
    device = torch.device(f'cuda:{device_id}')
    
    # 记录已分配的张量列表
    allocated_tensors = []
    
    print(f"Monitoring GPU memory on device {device_id}...\n")

    while True:
        # 获取当前 GPU 上的显存信息
        total_memory = torch.cuda.get_device_properties(device_id).total_memory
        reserved_memory = torch.cuda.memory_reserved(device_id)
        allocated_memory = torch.cuda.memory_allocated(device_id)
        free_memory = total_memory - reserved_memory - allocated_memory
        
        print(f"Total memory: {total_memory / 1e6:.2f} MB")
        print(f"Allocated memory: {allocated_memory / 1e6:.2f} MB")
        print(f"Reserved memory: {reserved_memory / 1e6:.2f} MB")
        print(f"Free memory: {free_memory / 1e6:.2f} MB\n")
        
        # 如果有空余显存，分配剩余的显存
        if free_memory > 0:
            try:
                block_size = free_memory // 4  # 每个 float32 元素占用 4 字节
                new_tensor = torch.empty((block_size,), dtype=torch.float32, device=device)
                allocated_tensors.append(new_tensor)
                print(f"Allocated additional tensor of size: {block_size} floats.")
            except RuntimeError as e:
                print(f"Failed to allocate memory: {e}")
        
        # 每一秒更新一次
        time.sleep(1)

if __name__ == "__main__":
    occupy_gpu_memory(device_id=0)  # 默认占用 GPU 0
