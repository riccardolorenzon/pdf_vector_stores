from .sql_memory import build_sql_memory
from .window_memory import build_window_buffer_memory 
memory_map = {
    "sql_memory": build_sql_memory, 
    "sql_window_memory": build_window_buffer_memory
}