# 外部计数器
_counter = 0


def get_next_id():
    global _counter
    _counter += 1
    return _counter
