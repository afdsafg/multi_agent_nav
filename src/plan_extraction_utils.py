import re
from typing import List, Dict, Optional

def extract_predictive_plan(planner_output: str) -> Optional[List[Dict[str, str]]]:
    """
    从high-level planner的输出中提取plan信息
    
    Args:
        planner_output: high-level planner的完整输出字符串
        
    Returns:
        List[Dict[str, str]]: 包含任务信息的字典列表，每个字典包含'task'和'status'键
        如果未找到plan则返回None
    """
    if not planner_output or not isinstance(planner_output, str):
        return None
    
    # 定义匹配XML格式todo list的正则表达式
    # 匹配 <update_todo_list><todos>...内容...</todos></update_todo_list>
    pattern = r'<update_todo_list>\s*<todos>(.*?)</todos>\s*</update_todo_list>'
    match = re.search(pattern, planner_output, re.DOTALL)
    
    if not match:
        # 如果没有找到XML格式，尝试直接匹配任务格式
        return extract_todo_list_from_text(planner_output)
    
    todos_content = match.group(1)
    
    # 提取每行任务，格式为 [状态] 任务描述
    todo_items = []
    lines = todos_content.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 匹配 [ ] 或 [-] 或 [x] 开头的任务
        task_match = re.match(r'^(\[[ x-]\])\s*(.+)$', line)
        if task_match:
            status = task_match.group(1).strip()
            task_description = task_match.group(2).strip()
            
            # 标准化状态表示
            if status == '[ ]':
                standardized_status = 'pending'
            elif status == '[-]':
                standardized_status = 'in_progress'
            elif status == '[x]':
                standardized_status = 'completed'
            else:
                standardized_status = 'unknown'
                
            if task_description:  # 只添加有描述的任务
                todo_items.append({
                    'task': task_description,
                    'status': standardized_status
                })
    
    return todo_items if todo_items else None


def extract_todo_list_from_text(text: str) -> Optional[List[Dict[str, str]]]:
    """
    从纯文本中直接提取todo list格式的任务
    
    Args:
        text: 包含todo list的文本
        
    Returns:
        List[Dict[str, str]]: 包含任务信息的字典列表
    """
    if not text or not isinstance(text, str):
        return None
    
    # 匹配 [ ] 或 [-] 或 [x] 开头的任务，跨多行
    todo_items = []
    
    # 按行分割文本
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 匹配 [状态] 任务描述格式
        task_match = re.match(r'^(\[[ x-]\])\s*(.+)$', line)
        if task_match:
            status = task_match.group(1).strip()
            task_description = task_match.group(2).strip()
            
            # 标准化状态表示
            if status == '[ ]':
                standardized_status = 'pending'
            elif status == '[-]':
                standardized_status = 'in_progress'
            elif status == '[x]':
                standardized_status = 'completed'
            else:
                standardized_status = 'unknown'
                
            if task_description:  # 只添加有描述的任务
                todo_items.append({
                    'task': task_description,
                    'status': standardized_status
                })
    
    return todo_items if todo_items else None


def get_tasks_by_status(todo_list: List[Dict[str, str]], status: str) -> List[Dict[str, str]]:
    if not todo_list:
        return []
    
    return [task for task in todo_list if task.get('status') == status]


def get_pending_tasks(todo_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return get_tasks_by_status(todo_list, 'pending')


def get_in_progress_tasks(todo_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return get_tasks_by_status(todo_list, 'in_progress')


def get_completed_tasks(todo_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return get_tasks_by_status(todo_list, 'completed')


def print_todo_list(todo_list: List[Dict[str, str]]) -> None:
    if not todo_list:
        print("No tasks found")
        return
    
    print("Todo List:")
    for i, task in enumerate(todo_list):
        status = task.get('status', 'unknown')
        task_desc = task.get('task', '')
        
        status_symbol = {
            'pending': '[ ]',
            'in_progress': '[-]',
            'completed': '[x]'
        }.get(status, '[?]')
        
        print(f"  {status_symbol} {task_desc}")
