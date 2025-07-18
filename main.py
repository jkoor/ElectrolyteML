import psutil
import time
import os


def get_memory_usage():
    """获取当前进程的内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    # 返回兆字节数
    return memory_info.rss / 1024 / 1024  # RSS: Resident Set Size


# 在关键点检查内存
print(f"初始内存: {get_memory_usage():.2f} MB")

from battery import Electrolyte
from material import Material

# 记录时间点
begin_time = time.time()

Electrolyte.from_jsons("database/calisol23.json")

# 记录时间点
end_time = time.time()
print(f"加载后内存: {get_memory_usage():.2f} MB")
print(f"加载耗时: {end_time - begin_time:.2f} 秒")
pass
