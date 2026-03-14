import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque
import time 
import os 
import math

# ==========================================
# 0. 固定隨機種子
# ==========================================
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)

# ==========================================
# 1. Global Constants & Topology
# ==========================================
LANE_OFFSET = 0.16   # 用於 N0-N7 (角落)
NODE_DIST = 0.35     # 外圈半徑
INNER_DIST = 0.18    # 內圈 N12-N15 的半徑

# 定義 16 個節點的連接圖
TILE_GRAPH = {
    0: [7, 8],  1: [8, 2],  2: [1, 9],  3: [9, 4],
    4: [3, 10], 5: [10, 6], 6: [5, 11], 7: [11, 0],
    8:  [0, 1, 12],   9:  [2, 3, 13],
    10: [4, 5, 14],   11: [6, 7, 15],
    12: [8, 15, 13],  13: [9, 12, 14],
    14: [10, 13, 15], 15: [11, 14, 12]
}

# [移到最上方] 取得 16 個節點的物理座標
def get_octagon_nodes(r, c):
    cx, cy = c, r
    # N0~N7
    n0 = (cx - LANE_OFFSET, cy - NODE_DIST)
    n1 = (cx + LANE_OFFSET, cy - NODE_DIST)
    n2 = (cx + NODE_DIST, cy - LANE_OFFSET)
    n3 = (cx + NODE_DIST, cy + LANE_OFFSET)
    n4 = (cx + LANE_OFFSET, cy + NODE_DIST)
    n5 = (cx - LANE_OFFSET, cy + NODE_DIST)
    n6 = (cx - NODE_DIST, cy + LANE_OFFSET)
    n7 = (cx - NODE_DIST, cy - LANE_OFFSET)
    # N8~N11
    n8  = (cx, cy - NODE_DIST)
    n9  = (cx + NODE_DIST, cy)
    n10 = (cx, cy + NODE_DIST)
    n11 = (cx - NODE_DIST, cy)
    # N12~N15
    n12 = (cx, cy - INNER_DIST)
    n13 = (cx + INNER_DIST, cy)
    n14 = (cx, cy + INNER_DIST)
    n15 = (cx - INNER_DIST, cy)
    return [n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15]

# [修正] 全域的 BFS 路徑搜尋 (供統計與繪圖使用)
def get_shortest_ring_path(start_idx, end_idx):
    if start_idx == end_idx: return [start_idx]
    queue = deque([(start_idx, [start_idx])])
    visited = {start_idx}
    while queue:
        curr, path = queue.popleft()
        if curr == end_idx: return path
        for neighbor in TILE_GRAPH[curr]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return []

# [輔助] 邊界名稱
def get_boundary_name(r, c, rows, cols):
    names = []
    if r == rows - 1: names.append(f"B{c}")
    if c == cols - 1: names.append(f"B{21 + (rows - 1 - r)}")
    if r == 0: names.append(f"B{42 + (cols - 1 - c)}")
    if c == 0: names.append(f"B{63 + r}")
    return "/".join(names) if names else "Unknown"

def get_boundary_phys_coords(r, c, rows, cols):
    if r == rows - 1: return (rows - 0.2, c)
    if c == cols - 1: return (r, cols - 0.2)
    if r == 0: return (-0.8, c)
    if c == 0: return (r, -0.8)
    return (r, c)

def get_closest_perimeter_index(pin_r, pin_c, perimeter_list, rows, cols):
    best_idx = -1
    min_dist = float('inf')
    for i, (br, bc) in enumerate(perimeter_list):
        b_phys = get_boundary_phys_coords(br, bc, rows, cols)
        dist = abs(pin_r - b_phys[0]) + abs(pin_c - b_phys[1])
        if dist < min_dist:
            min_dist = dist
            best_idx = i
    return best_idx

def get_best_candidate_indices_circular(pin_r, pin_c, perimeter_list, cursor, usage_map, cap_per_tile, rows, cols):
    total_len = len(perimeter_list)
    candidates = []
    w_phys_dist = 1.0; w_gap_dist = 0.8    
    for i in range(total_len):
        idx = (cursor + i) % total_len
        if usage_map.get(idx, 0) >= cap_per_tile: continue
        br, bc = perimeter_list[idx]
        b_phys = get_boundary_phys_coords(br, bc, rows, cols)
        phys_dist = abs(pin_r - b_phys[0]) + abs(pin_c - b_phys[1])
        gap_dist = i 
        cost = (w_phys_dist * phys_dist) + (w_gap_dist * gap_dist)
        candidates.append({'idx': idx, 'cost': cost})
    candidates.sort(key=lambda x: x['cost'])
    return [c['idx'] for c in candidates]

class PCBRouterD3QN(nn.Module):
    def __init__(self, grid_h, grid_w, action_space_size):
        super(PCBRouterD3QN, self).__init__()
        
        # --- 卷積層 (特徵提取) ---
        # 輸入通道數為 6 (對應環境的 obs, pos, v_lock, h_lock, valid_b, future_map)
        self.conv = nn.Sequential(
            # 第一層：stride=1，保持解析度 (20x20 -> 20x20)，提取精準的 Pin 與障礙物位置
            nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            # 第二層：stride=2，第一次降維 (20x20 -> 10x10)，提取局部路徑結構
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # 第三層：stride=2，第二次降維 (10x10 -> 5x5)，提取全局擁擠度與大方向
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            nn.Flatten()
        )
        
        # --- 動態計算 Flatten 後的維度 ---
        # 因為我們總共用了兩次 stride=2 的卷積層，長寬各會被縮小 4 倍 (2 * 2 = 4)
        # 對於 20x20 的網格：ceil(20/4) = 5 -> 最終輸出為 128 * 5 * 5 = 3200
        conv_h = math.ceil(grid_h / 4)
        conv_w = math.ceil(grid_w / 4)
        flat_dim = 128 * conv_h * conv_w 
        
        # --- Dueling Network 雙流架構 ---
        
        # 1. Advantage Stream (計算各個動作的相對優勢 A(s,a))
        self.fc_advantage = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_space_size)
        )
        
        # 2. Value Stream (計算當前狀態的絕對價值 V(s))
        self.fc_value = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # 輸出固定為 1 維的標量
        )


    def forward(self, x):
        x = self.conv(x)
        
        advantage = self.fc_advantage(x)
        value = self.fc_value(x)
        
        # Dueling 合併公式: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        # 這樣做是為了確保 V(s) 能代表狀態的平均價值
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

# ==========================================
# 2. PCB 環境
# ==========================================

# 新增在 class PCBGridEnv 外面，或作為靜態方法
def find_disjoint_paths(graph, s1, t1, s2, t2, max_paths=20):
    """
    在 Tile Graph 中尋找兩條不相交的路徑。
    s1, t1: 舊 Net 的起點與終點
    s2, t2: 新 Net (當前 Agent) 的起點與終點
    max_paths: 限制 DFS 最多只找幾條備用路徑，防止 CPU 運算爆炸
    回傳: (path1, path2) 如果成功，否則回傳 None
    """
    
    # 1. 找出 Net 1 (舊線) 的備用路徑 (加上數量限制的 DFS)
    all_paths_1 = []
    
    def dfs(current_node, end_node, visited_mask, current_path):
        # 🔥 煞車機制：只要找到 max_paths 條路徑，就直接停止搜尋，解放 CPU
        if len(all_paths_1) >= max_paths:
            return
            
        if current_node == end_node:
            all_paths_1.append(current_path)
            return
            
        for neighbor in graph[current_node]:
            if neighbor not in visited_mask:
                dfs(neighbor, end_node, visited_mask | {neighbor}, current_path + [neighbor])

    # 啟動 DFS 搜尋
    dfs(s1, t1, {s1}, [s1])
    
    # 根據長度排序，優先嘗試讓 Net 1 走較短的路
    all_paths_1.sort(key=len)

    # 2. 針對 Net 1 的每一種走法，嘗試幫 Net 2 找路
    for p1 in all_paths_1:
        p1_set = set(p1)
        
        # 如果 s2 或 t2 已經被 p1 佔用，這組 p1 無效
        if s2 in p1_set or t2 in p1_set:
            continue
            
        # 在 "排除 p1 節點" 的圖中尋找 Net 2 的路徑 (只找一條最短的即可，用 BFS)
        queue = deque([(s2, [s2])])
        visited = {s2}
        p2_found = None
        
        while queue:
            curr, path = queue.popleft()
            if curr == t2:
                p2_found = path
                break
            
            for neighbor in graph[curr]:
                if neighbor not in visited and neighbor not in p1_set:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        if p2_found:
            return p1, p2_found # 成功找到雙贏解！

    return None # 20 種繞法都行不通，無解
class PCBGridEnv:
    def __init__(self, rows=21, cols=21, capacity=4, pins=None):
        self.rows = rows
        self.cols = cols
        self.initial_capacity = capacity
        self.primary_target_pos = None 
        self.pins = pins 
        self.perimeter_list = generate_perimeter_path(rows, cols) 
        self.current_candidate_indices = [] 
        self.capacity_map = np.full((rows, cols), capacity)
        self.direction_map = np.zeros((rows, cols), dtype=int) 
        self.sub_lane_map = np.zeros((rows, cols, 4), dtype=bool)
        # [新增] 用來儲存所有已完成 Net 的詳細路徑，讓 env 可以隨時修改它
        # 格式: { net_id: (path_coords, path_lanes, escape_trace, node_seqs) }
        self.commited_paths = {}
        # [NEW] 邊界 Via 佔用表 (取代原本的隱形牆與互斥邏輯)
        # Key: (r, c, lane_idx), Value: net_id
        self.boundary_via_map = {} 
        self.blocked_moves = set()
        self.node_occupancy = np.zeros((rows, cols, 16), dtype=int)
        self.center_map = np.zeros((rows, cols, 2), dtype=bool)
        self.current_node_sequences = []
        self.my_lane_idx = 0 
        self.current_entry_node_idx = 0
        self.current_target_mask = np.zeros((rows, cols), dtype=bool)
        self.current_path = []
        self.current_path_lanes = [] 
        self.head_pos = (0, 0)
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
        self.escape_trace = [] 
        self.start_entry_type = None 
        self.current_net_id = 0 
        self.is_direct_done = False 

    # ... (get_region, _get_shortest_ring_path 等函式保持不變) ...
    def get_region(self, idx):
        if 0 <= idx < self.cols: return 0
        if self.cols <= idx < self.cols + self.rows: return 1
        if self.cols + self.rows <= idx < 2 * self.cols + self.rows: return 2
        return 3
    
    # [NEW] 檢查黃色 Via 是否被擋住
    def _is_via_blocked(self, r, c, lane_idx, my_net_id):
        key = (r, c, lane_idx)
        occupier = self.boundary_via_map.get(key, 0)
        # 如果有人佔用 (!=0) 且那個人不是我 (!=my_net_id)，那就是擋住了
        if occupier != 0 and occupier != my_net_id:
            return True 
        return False

    # [NEW] 設定佔用 (在黃色 Via 插旗子)
    def _occupy_via(self, r, c, lane_idx, my_net_id):
        key = (r, c, lane_idx)
        self.boundary_via_map[key] = my_net_id

    
    def _get_exit_node_index(self, curr_r, curr_c, next_r, next_c, curr_lane, next_lane):
        """
        curr_lane: 進入當前 Tile 的車道 (從上一格傳進來的)
        next_lane: 準備離開當前 Tile，跨越邊界進入下一格要用的車道
        """
        dr, dc = next_r - curr_r, next_c - curr_c
        
        # 出口節點必須與「即將跨越的邊界」以及「下一格要對齊的軌道(next_lane)」一致
        if dr == -1: # 向上移動 (Up)
            # 只有在垂直移動時，next_lane 才會是 2 或 3
            return 1 if next_lane == 3 else 0
        elif dr == 1: # 向下移動 (Down)
            return 5 if next_lane == 2 else 4
        elif dc == -1: # 向左移動 (Left)
            # 只有在水平移動時，next_lane 才會是 0 或 1
            return 7 if next_lane == 0 else 6
        elif dc == 1: # 向右移動 (Right)
            return 3 if next_lane == 1 else 2
            
        return 0
    
    # 新增在 class PCBGridEnv 內部
    # 修正後的協商邏輯 (修復了缺少 _infer_entry_exit 的問題)
    # 修正後的協商邏輯 (修復了缺少 _infer_entry_exit 的問題)
    def try_resolve_conflict_in_tile(self, r, c, new_net_id, old_net_id, new_entry, new_exit):
        """
        修正版：針對十字交錯與連環追撞問題，加入嚴格的重疊檢查。
        """
        # print(f"\n{'='*20} 協商啟動: Tile ({r}, {c}) {'='*20}")
        
        old_entry, old_exit = -1, -1
        
        # 1. 精準鎖定舊線路 (Double Check Coordinates)
        found_segment = False
        if old_net_id in self.commited_paths:
            p_coords, _, _, p_seqs = self.commited_paths[old_net_id]
            for i, (pr, pc) in enumerate(p_coords):
                # 強制轉型 int 比較，避免 float 誤差
                if int(pr) == int(r) and int(pc) == int(c):
                    if i < len(p_seqs) and p_seqs[i]:
                        old_entry = p_seqs[i][0]
                        old_exit = p_seqs[i][-1]
                        found_segment = True
                        break # 找到這格的紀錄就跳出
        
        # [關鍵] 如果找不到確切紀錄，或者找到的紀錄看起來怪怪的，直接放棄
        if not found_segment or old_entry == -1:
            # print(f"  [保護] 無法確認 P{old_net_id} 在此格的原始流向，拒絕協商以免斷路。")
            return None

        # print(f"  [鎖定] P{old_net_id}: {old_entry}->{old_exit} | P{new_net_id}: {new_entry}->{new_exit}")

        # 2. 進行路徑搜尋 (這時候如果是十字交錯，這裡就會回傳 None)
        result = find_disjoint_paths(TILE_GRAPH, old_entry, old_exit, new_entry, new_exit)
        
        if result:
            p_old_new, p_new_new = result
            
            # -------------------------------------------------------------------
            # [新增防護 1] 嚴格檢查兩條新路徑是否真的不相交 (修復 P32/P33 互撞)
            # -------------------------------------------------------------------
            set_old = set(p_old_new)
            set_new = set(p_new_new)
            if not set_old.isdisjoint(set_new):
                # print(f"  [協商無效] 演算法產生了重疊路徑: {set_old & set_new}，放棄協商。")
                return None
            
            # [雙重保險] 確認演算法沒有偷改 P32 的頭尾
            if p_old_new[0] != old_entry or p_old_new[-1] != old_exit:
                # print(f"  [嚴重錯誤] 協商結果破壞了 P{old_net_id} 的連通性。")
                return None

            # -------------------------------------------------------------------
            # [新增防護 2] 檢查新路徑是否會撞到「第三者」(修復 P32 蓋掉 P31)
            # -------------------------------------------------------------------
            # 我們要檢查 p_new_new (P33要走的路) 和 p_old_new (P32要改走的路)
            # 是否會撞到 Tile 內其他既有的 Net (例如 P31)
            
            # 檢查 P32 的新家是否安全
            for n_idx in p_old_new:
                occupier = self.node_occupancy[r, c, n_idx]
                # 如果這格有人，且不是 P32 自己，也不是 P33 (我們要搶的)，那就是無辜路人
                if occupier != 0 and occupier != old_net_id and occupier != new_net_id:
                    # print(f"  [協商無效] P{old_net_id} 的新路徑會撞到 P{occupier}，放棄。")
                    return None
            
            # 檢查 P33 的新家是否安全
            for n_idx in p_new_new:
                occupier = self.node_occupancy[r, c, n_idx]
                if occupier != 0 and occupier != old_net_id and occupier != new_net_id:
                    # print(f"  [協商無效] P{new_net_id} 的新路徑會撞到 P{occupier}，放棄。")
                    return None

            # -------------------------------------------------------------------
            # 3. 物理層更新 (只有在真的成功且安全時才做)
            # -------------------------------------------------------------------
            # print(f"  [結果] ✅ 協商成功！物理層已更新。")
            
            # 清除舊路徑佔用
            for n_idx in range(16):
                if self.node_occupancy[r, c, n_idx] == old_net_id:
                    self.node_occupancy[r, c, n_idx] = 0
            
            # 寫入舊 Net 的新路徑
            for n_idx in p_old_new:
                self.node_occupancy[r, c, n_idx] = old_net_id
            
            # 更新歷史紀錄 (這一步很重要，不然下次 P32 被別人協商時會找不到路)
            self._update_net_history(old_net_id, r, c, p_old_new)
            
            # 回傳新 Net 的路徑 (外部函式會負責將其寫入 node_occupancy)
            return p_new_new
            
        else:
            # 這才是十字交錯該有的結局：失敗
            # print(f"  [結果] ❌ 協商失敗 (拓樸衝突/無解)。")
            return None
    
    def _get_net_segment_entry_exit(self, net_id, r, c):
        """
        查詢特定 Net 在指定 Tile (r, c) 的入口與出口節點 index。
        回傳: (entry_node, exit_node) 或 None (若找不到資料)
        """
        if net_id not in self.commited_paths:
            return None
            
        data = self.commited_paths[net_id]
        # 相容性處理 (你的程式碼中有兩種 tuple 長度)
        if len(data) == 4:
            path_coords, _, _, node_seqs = data
        else:
            # 如果是舊格式沒有 node_seqs，無法判斷詳細流向，視為失敗
            return None

        # 找到 (r, c) 在該 Net 路徑中的位置
        for i, (pr, pc) in enumerate(path_coords):
            if int(pr) == int(r) and int(pc) == int(c):
                if i < len(node_seqs) and node_seqs[i]:
                    # 該 Tile 的路徑序列：第一個是入口，最後一個是出口
                    return node_seqs[i][0], node_seqs[i][-1]
        
        return None
    
    def _get_entry_node_index(self, prev_exit_idx):
        mapping = { 4: 1, 5: 0, 0: 5, 1: 4, 2: 7, 3: 6, 6: 3, 7: 2 }
        return mapping.get(prev_exit_idx, 0)
    
    # [修改] 取得最近節點 (因為多了內圈，要比對 16 個點)
    def _get_closest_node_index(self, r, c, phys_r, phys_c):
        nodes = get_octagon_nodes(r, c)
        best_idx = -1
        min_dist = float('inf')
        
        # 為了避免外部 Pin 直接連到內圈 (N12-N15)，我們可以只檢查外圈 N0-N11
        # 或者如果您希望 Pin 可以直接穿透到內圈，就檢查 range(16)
        # 建議：Pin 通常在外圍，所以優先連到外圈
        check_indices = range(12) # 只檢查 N0 ~ N11
        
        for i in check_indices:
            nx, ny = nodes[i] # get_octagon_nodes 回傳的是 (x, y) = (col, row)
            dist = math.sqrt((nx - phys_c)**2 + (ny - phys_r)**2)
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        return best_idx

    # [修改] 新增 current_cursor 參數，預設為 0
    # [修改] 移除車道檢查，只保留游標順序檢查
    # [MODIFIED] 核心修正：並行檢查多個跳脫方向，並放寬邊界判定
    # [MODIFIED] 核心修正：包含 Direct Escape 與 方向性起點偏好 (Directional Start Preference)
    # [MODIFIED] 核心修正：包含 Direct Escape 與方向性起點偏好，並加入容錯檢查邏輯
    def get_possible_starts(self, pin_idx, target_phys_pos, current_cursor=0, debug=False):
        pin_data = self.pins[pin_idx]
        pr, pc = pin_data[1], pin_data[2]
        tr, tc = target_phys_pos

        # 1. 定義物理邊界區域
        is_bottom_zone = (pr >= self.rows - 1.5)
        is_top_zone    = (pr <= 0.6)
        is_right_zone  = (pc >= self.cols - 1.5)
        is_left_zone   = (pc <= 0.6)
        
        is_boundary_pin = is_bottom_zone or is_top_zone or is_right_zone or is_left_zone

        # =============================================================
        # 嘗試 1: Direct Escape (優先嘗試直接跳脫)
        # =============================================================
        if is_boundary_pin:
            direct_candidates = []
            cursor_is_empty = False
            if 0 <= current_cursor < len(self.perimeter_list):
                cr_tile, cc_tile = self.perimeter_list[current_cursor]
                if self.capacity_map[cr_tile, cc_tile] == self.initial_capacity:
                    cursor_is_empty = True

            if is_bottom_zone: 
                b_idx = int(pc)
                if b_idx >= current_cursor or ((b_idx + 1) == current_cursor and cursor_is_empty):
                    dist = abs(pr - (self.rows - 0.2))
                    direct_candidates.append({'r': int(pr), 'c': int(pc), 'via': (self.rows - 0.2, pc), 'lane_idx': -1, 'node_pos': (self.rows - 0.2, pc), 'score': dist * 0.1})

            if is_top_zone:
                b_idx = 42 + (self.cols - 1 - int(pc))
                if b_idx >= current_cursor or ((b_idx + 1) == current_cursor and cursor_is_empty):
                    dist = abs(pr - (-0.8))
                    direct_candidates.append({'r': int(pr), 'c': int(pc), 'via': (-0.8, pc), 'lane_idx': -1, 'node_pos': (-0.8, pc), 'score': dist * 0.1})

            if is_right_zone:
                b_idx = 21 + (self.rows - 1 - int(pr))
                if b_idx >= current_cursor or ((b_idx + 1) == current_cursor and cursor_is_empty):
                    dist = abs(pc - (self.cols - 0.2))
                    direct_candidates.append({'r': int(pr), 'c': int(pc), 'via': (pr, self.cols - 0.2), 'lane_idx': -1, 'node_pos': (pr, self.cols - 0.2), 'score': dist * 0.1})

            if is_left_zone:
                b_idx = 63 + int(pr)
                if b_idx >= current_cursor or ((b_idx + 1) == current_cursor and cursor_is_empty):
                    dist = abs(pc - (-0.8))
                    direct_candidates.append({'r': int(pr), 'c': int(pc), 'via': (pr, -0.8), 'lane_idx': -1, 'node_pos': (pr, -0.8), 'score': dist * 0.1})

            if direct_candidates:
                if debug: print(f"   -> [Direct] 找到 {len(direct_candidates)} 個直接跳脫點")
                final_candidates = []
                for dc in direct_candidates:
                     final_candidates.append({
                        'start_info': (dc['r'], dc['c'], pr, pc, 'DIRECT', -1, dc['node_pos']),
                        'score': dc['score'],
                        'debug_node_idx': -99
                    })
                final_candidates.sort(key=lambda x: x['score'])
                return [c['start_info'] for c in final_candidates]

        # =============================================================
        # 嘗試 2: Standard DQN Routing (標準尋路) - Fallback
        # =============================================================
        if debug and is_boundary_pin:
            print(f"   -> [Fallback] Direct Escape 失敗，切換至 DQN 模式")

        node_map = {
            0: {'lane': 2, 'type': 'V', 'side': 'left'},  
            1: {'lane': 3, 'type': 'V', 'side': 'right'}, 
            2: {'lane': 0, 'type': 'H', 'side': 'top'},    
            3: {'lane': 1, 'type': 'H', 'side': 'bottom'},
            4: {'lane': 3, 'type': 'V', 'side': 'right'}, 
            5: {'lane': 2, 'type': 'V', 'side': 'left'},  
            6: {'lane': 1, 'type': 'H', 'side': 'bottom'},
            7: {'lane': 0, 'type': 'H', 'side': 'top'},    
        }

        r_base, c_base = int(np.floor(pr)), int(np.floor(pc))
        
        candidate_info = [
            {'r': r_base, 'c': c_base, 'allowed': [3, 4], 'pos': 'TL'},        
            {'r': r_base, 'c': c_base + 1, 'allowed': [5, 6], 'pos': 'TR'}, 
            {'r': r_base + 1, 'c': c_base, 'allowed': [1, 2], 'pos': 'BL'}, 
            {'r': r_base + 1, 'c': c_base + 1, 'allowed': [0, 7], 'pos': 'BR'} 
        ]

        target_region = None
        if tr >= self.rows - 1.0: target_region = 'BOTTOM'
        elif tr <= 0.8: target_region = 'TOP'
        elif tc >= self.cols - 1.0: target_region = 'RIGHT'
        elif tc <= 0.8: target_region = 'LEFT'

        valid_tiles = []
        for item in candidate_info:
            r, c = item['r'], item['c']
            if 0 <= r < self.rows and 0 <= c < self.cols:
                tile_center_r, tile_center_c = r + 0.5, c + 0.5
                dist_to_target = math.sqrt((tile_center_r - tr)**2 + (tile_center_c - tc)**2)
                valid_tiles.append({'r': r, 'c': c, 'allowed': item['allowed'], 'pos': item['pos'], 'dist': dist_to_target})

        valid_tiles.sort(key=lambda x: x['dist'])
        final_candidates = []

        for tile in valid_tiles:
            t_r, t_c = tile['r'], tile['c']
            t_pos = tile['pos']
            allowed_nodes = tile['allowed'] 
            all_octagon_nodes = get_octagon_nodes(t_r, t_c)
            tile_center_r, tile_center_c = t_r + 0.5, t_c + 0.5
            target_is_vertical = abs(tr - tile_center_r) > abs(tc - tile_center_c)

            for node_idx in allowed_nodes:
                # [FIXED] 正確解包：get_octagon_nodes 回傳 (x, y) = (Col, Row)
                n_col, n_row = all_octagon_nodes[node_idx]
                
                info = node_map[node_idx]
                lane_type = info['type']
                lane_idx = info['lane']
                
                if lane_type == 'H':
                    via_phys = (n_row, pc) 
                else:
                    via_phys = (pr, n_col) 

                occupier = self.node_occupancy[t_r, t_c, node_idx]
                if occupier != 0 and occupier != self.current_net_id: continue
                if self.capacity_map[t_r, t_c] <= 0: continue

                # [FIXED] 距離公式修正
                dist_node_to_target = math.sqrt((n_row - tr)**2 + (n_col - tc)**2)
                
                dir_penalty = 0.0
                if target_is_vertical and lane_type == 'H': dir_penalty = 2.0
                elif not target_is_vertical and lane_type == 'V': dir_penalty = 2.0
                
                pref_bonus = 0.0
                if target_region == 'BOTTOM':
                    if t_pos == 'BL' and node_idx == 2: pref_bonus = -5.0
                    elif t_pos == 'BR' and node_idx == 7: pref_bonus = -5.0
                elif target_region == 'RIGHT':
                    if t_pos == 'TR' and node_idx == 5: pref_bonus = -5.0
                    elif t_pos == 'BR' and node_idx == 0: pref_bonus = -5.0
                elif target_region == 'TOP':
                    if t_pos == 'TL' and node_idx == 3: pref_bonus = -5.0
                    elif t_pos == 'TR' and node_idx == 6: pref_bonus = -5.0
                elif target_region == 'LEFT':
                    if t_pos == 'TL' and node_idx == 4: pref_bonus = -5.0
                    elif t_pos == 'BL' and node_idx == 1: pref_bonus = -5.0

                # 方向懲罰 (Wrong Side Penalty)
                wrong_side_penalty = 0.0
                if target_region == 'BOTTOM' or target_region == 'TOP':
                    # 如果目標在左 (tc < pc) 但格子在右 (n_col > pc) -> 懲罰
                    if tc < pc and n_col > pc: wrong_side_penalty = 5.0
                    # 如果目標在右 (tc > pc) 但格子在左 (n_col < pc) -> 懲罰
                    elif tc > pc and n_col < pc: wrong_side_penalty = 5.0
                
                elif target_region == 'RIGHT' or target_region == 'LEFT':
                    # 如果目標在上 (tr < pr) 但格子在下 (n_row > pr) -> 懲罰
                    if tr < pr and n_row > pr: wrong_side_penalty = 5.0
                    # 如果目標在下 (tr > pr) 但格子在上 (n_row < pr) -> 懲罰
                    elif tr > pr and n_row < pr: wrong_side_penalty = 5.0

                total_score = dist_node_to_target + dir_penalty + 5.0 + pref_bonus + wrong_side_penalty
                
                # [Removed] P4 Debug Print

                final_candidates.append({
                    'start_info': (t_r, t_c, via_phys[0], via_phys[1], 
                                   lane_type, lane_idx, (n_row, n_col)), 
                    'score': total_score,
                    'debug_node_idx': node_idx
                })

            if len(final_candidates) > 4: break
        
        final_candidates.sort(key=lambda x: x['score'])
        return [c['start_info'] for c in final_candidates]
    def _is_straight_path(self, entry_node, exit_node):
        """判斷是否為穿過中心的直行路徑"""
        # 對角線或對面節點的組合視為直行
        # 0(TL)-5(BL), 1(TR)-4(BR) -> 垂直直行
        # 7(TL)-2(TR), 6(BL)-3(BR) -> 水平直行
        
        # 簡單判斷：如果距離為 4 (對面) 或特殊的平行對邊
        # 這裡我們列舉所有「會切過中心」的組合
        straight_pairs = [
            {0, 5}, {5, 0}, # Lane 2 (Vertical)
            {1, 4}, {4, 1}, # Lane 3 (Vertical)
            {2, 7}, {7, 2}, # Lane 0 (Horizontal)
            {3, 6}, {6, 3}  # Lane 1 (Horizontal)
        ]
        return {entry_node, exit_node} in straight_pairs
    
    def prepare_pin_routing(self, target_mask, net_id, candidate_indices):
        self.current_target_mask = target_mask
        self.current_net_id = net_id 
        self.current_candidate_indices = candidate_indices
        if candidate_indices and len(candidate_indices) > 0:
            best_idx = candidate_indices[0]
            self.primary_target_pos = self.perimeter_list[best_idx]
        else:
            self.primary_target_pos = None

    def _calculate_lookahead_reward(self, current_pos, next_pos):
        if not self.current_candidate_indices or self.current_net_id >= len(self.pins): return 0.0
        min_idx = min(self.current_candidate_indices)
        max_idx = max(self.current_candidate_indices)
        reg_min = self.get_region(min_idx)
        reg_max = self.get_region(max_idx)
        r_min_b, c_min_b = self.perimeter_list[min_idx]
        r_max_b, c_max_b = self.perimeter_list[max_idx]
        cr, cc = current_pos; nr, nc = next_pos
        pin_data = self.pins[self.current_net_id - 1]
        pin_r, pin_c = pin_data[1], pin_data[2]
        future_pins = self.pins[self.current_net_id:] 
        bonus = 0.0; penalty = 0.0; reward_val = 3.0; penalty_val = -6.0

        if reg_min == 0 and reg_max == 0:
            c_start = min(c_min_b, c_max_b)
            c_end = max(c_min_b, c_max_b, pin_c)
            rect_pins = [p for p in future_pins if p[1] >= pin_r and c_start <= p[2] <= c_end]
            if rect_pins:
                target_val = min(p[2] for p in rect_pins)
                if cc > target_val and nc < cc: bonus += reward_val
                if nc < target_val: bonus += (reward_val * 2.0)
                elif nc > target_val: penalty += penalty_val
        elif reg_min == 0 and reg_max == 1:
            r_start = min(r_max_b, pin_r) 
            c_start = min(c_min_b, pin_c)
            rect_pins = [p for p in future_pins if p[1] >= r_start and p[2] >= c_start]
            if rect_pins:
                min_c = min(p[2] for p in rect_pins)
                if cc > min_c and nc < cc: bonus += reward_val
                if nc < min_c: bonus += (reward_val * 2.0)
                elif nc > min_c: penalty += penalty_val
        elif reg_min == 1 and reg_max == 1:
            r_start = min(r_min_b, r_max_b, pin_r)
            r_end = max(r_min_b, r_max_b)
            rect_pins = [p for p in future_pins if r_start <= p[1] <= r_end and p[2] >= pin_c]
            if rect_pins:
                max_r = max(p[1] for p in rect_pins)
                if cr < max_r and nr > cr: bonus += reward_val
                if nr > max_r: bonus += (reward_val * 2.0)
                elif nr < max_r: penalty += penalty_val
        elif reg_min == 1 and reg_max == 2:
            r_end = max(r_min_b, pin_r)
            c_start = min(c_max_b, pin_c)
            rect_pins = [p for p in future_pins if p[1] <= r_end and p[2] >= c_start]
            if rect_pins:

                max_r = max(p[1] for p in rect_pins)
                if cr < max_r and nr > cr: bonus += reward_val
                if nr > max_r: bonus += (reward_val * 2.0)
                elif nr < max_r: penalty += penalty_val
        elif reg_min == 2 and reg_max == 2:
            c_start = min(c_min_b, c_max_b, pin_c)
            c_end = max(c_min_b, c_max_b)
            rect_pins = [p for p in future_pins if p[1] <= pin_r and c_start <= p[2] <= c_end]
            if rect_pins:
                max_c = max(p[2] for p in rect_pins)
                if cc < max_c and nc > cc: bonus += reward_val
                if nc > max_c: bonus += (reward_val * 2.0)
                elif nc < max_c: penalty += penalty_val
        elif reg_min == 2 and reg_max == 3:
            r_end = max(r_max_b, pin_r)
            c_end = max(c_min_b, pin_c)
            rect_pins = [p for p in future_pins if p[1] <= r_end and p[2] <= c_end]
            if rect_pins:
                max_c = max(p[2] for p in rect_pins)
                if cc < max_c and nc > cc: bonus += reward_val
                if nc > max_c: bonus += (reward_val * 2.0)
                elif nc < max_c: penalty += penalty_val
        elif reg_min == 3 and reg_max == 3:
            r_start = min(r_min_b, r_max_b)
            r_end = max(r_min_b, r_max_b, pin_r)
            rect_pins = [p for p in future_pins if r_start <= p[1] <= r_end and p[2] <= pin_c]
            if rect_pins:
                min_r = min(p[1] for p in rect_pins)
                if cr > min_r and nr < cr: bonus += reward_val
                if nr < min_r: bonus += (reward_val * 2.0)
                elif nr > min_r: penalty += penalty_val
        elif reg_min == 0 and reg_max == 3:
            r_start = min(r_max_b, pin_r)
            c_end = max(c_min_b, pin_c)
            rect_pins = [p for p in future_pins if p[1] >= r_start and p[2] <= c_end]
            if rect_pins:
                min_r = min(p[1] for p in rect_pins)
                if cr > min_r and nr < cr: bonus += reward_val
                if nr < min_r: bonus += (reward_val * 2.0)
                elif nr > min_r: penalty += penalty_val
        return bonus + penalty

    def set_start_tile(self, start_info, pin_center_vis, debug=False):
        self.current_node_sequences = []
        self.last_action = None 
        self.is_direct_done = False 
        self.path_action_history = set()
        # [修正] 初始化這個列表，避免 step() 報錯 AttributeError
        self.current_path_entries = [] 

        r, c, via_r, via_c, e_type, req_lane, node_phys = start_info
        
        # [Fix] 交換順序：傳入 node_phys[1] (row) 給 phys_r，node_phys[0] (col) 給 phys_c
        closest_node_idx = self._get_closest_node_index(r, c, node_phys[0], node_phys[1])

        # === Case 1: Direct Escape ===
        # === Case 1: Direct Escape ===
        # === Case 1: Direct Escape ===
        if e_type == 'DIRECT':
            self.my_lane_idx = -1 
            self.head_pos = (r, c)
            self.current_path = [(r, c)]
            self.escape_trace = [pin_center_vis, pin_center_vis, node_phys]
            
            self.current_path_lanes = [-1] 
            
            # 保持起點 Node 暢通
            self.current_entry_node_idx = closest_node_idx
            self.current_path_entries = [closest_node_idx]
            
            # =========================================================
            # [修正重點] 依照您指定的單側阻擋邏輯
            # =========================================================
            
            # --- 1. Top Escape (往上跳脫) ---
            # 案例 P21: Tile (0, 19)
            # 需求: 只擋住 (0, 19) 與 (0, 20) 之間的牆
            if r <= 1: 
                b_row = 0
                curr_r, curr_c = b_row, int(c)
                
                # 1. 自己不能往右 (Action 3)
                self.blocked_moves.add((curr_r, curr_c, 3))
                
                # 2. 右邊鄰居不能往左 (Action 2)
                if curr_c + 1 < self.cols:
                    self.blocked_moves.add((curr_r, curr_c + 1, 2))

            # --- 2. Bottom Escape (往下跳脫) ---
            # 案例 P11: Tile (20, 18)
            # 需求: 只擋住 (20, 18) 與 (20, 19) 之間的牆 (右側牆)
            elif r >= self.rows - 2: 
                b_row = self.rows - 1
                curr_r, curr_c = b_row, int(c)
                
                # 1. 自己不能往右 (Action 3)
                self.blocked_moves.add((curr_r, curr_c, 3))
                
                # 2. 右邊鄰居不能往左 (Action 2)
                if curr_c + 1 < self.cols:
                    self.blocked_moves.add((curr_r, curr_c + 1, 2))

            # --- 3. Right Escape (往右跳脫) ---
            # 案例 P12: Tile (17, 20)
            # 需求: 只擋住 (17, 20) 與 (18, 20) 之間的牆 (下方牆)
            elif c >= self.cols - 2: 
                b_col = self.cols - 1
                curr_r, curr_c = int(r), b_col
                
                # 1. 自己不能往下 (Action 1)
                self.blocked_moves.add((curr_r, curr_c, 1))
                
                # 2. 下方鄰居不能往上 (Action 0)
                if curr_r + 1 < self.rows:
                    self.blocked_moves.add((curr_r + 1, curr_c, 0))

            # --- 4. Left Escape (往左跳脫) ---
            # 案例 P41: Tile (19, 0)
            # 需求: 只擋住 (19, 0) 與 (20, 0) 之間的牆 (下方牆)
            elif c <= 1: 
                b_col = 0
                curr_r, curr_c = int(r), b_col
                
                # 1. 自己不能往下 (Action 1)
                self.blocked_moves.add((curr_r, curr_c, 1))
                
                # 2. 下方鄰居不能往上 (Action 0)
                if curr_r + 1 < self.rows:
                    self.blocked_moves.add((curr_r + 1, curr_c, 0))

            self.is_direct_done = True
            return self.get_state()

        # === Case 2: Normal Path ===
        self.head_pos = (r, c)
        self.current_path = [(r, c)]
        self.escape_trace = [pin_center_vis, (via_r, via_c), node_phys]
        self.my_lane_idx = req_lane
        self.current_path_lanes = [self.my_lane_idx]
        self.start_entry_type = e_type 
        
        # [修正] 統一使用前面計算好的 index
        self.current_entry_node_idx = closest_node_idx
        self.node_occupancy[r, c, self.current_entry_node_idx] = self.current_net_id
        
        # [修正] 這裡是導致您報錯的關鍵！必須初始化這個列表
        self.current_path_entries = [closest_node_idx]

        return self.get_state()

    def get_state(self):
        # 1. 障礙物地圖 (Obstacles)
        obs_map = np.zeros((self.rows, self.cols), dtype=np.float32)
        shared_mask = (self.capacity_map < self.initial_capacity) & (self.capacity_map > 0)
        obs_map[shared_mask] = 0.5
        full_mask = self.capacity_map <= 0
        obs_map[full_mask] = 1.0
        # 將當前路徑視為障礙，避免回頭
        for r, c in self.current_path: 
            obs_map[r, c] = 1.0 
            
        # 2. 當前位置地圖 (Current Position)
        pos_map = np.zeros((self.rows, self.cols), dtype=np.float32)
        pos_map[self.head_pos] = 1.0
        
        # 3. 方向鎖定地圖 (Direction Locks)
        v_lock = (self.direction_map == 1).astype(np.float32)
        h_lock = (self.direction_map == 2).astype(np.float32)
        
        # ==========================================
        # [修改核心] 4. 目標梯度地圖 (Target Distance Gradient Map)
        # 不再只給絕對座標亮點，而是給予整張地圖的「引力梯度」
        # ==========================================
        gradient_map = np.zeros((self.rows, self.cols), dtype=np.float32)
        target_coords = np.argwhere(self.current_target_mask) # 找出所有目標點 (r, c)
        
        if len(target_coords) > 0:
            # 建立 20x20 的網格座標矩陣
            grid_r, grid_c = np.indices((self.rows, self.cols))
            min_dist = np.full((self.rows, self.cols), fill_value=np.inf)
            
            # 計算地圖上每個點到目標的最短 Manhattan 距離
            for tr, tc in target_coords:
                dist = np.abs(grid_r - tr) + np.abs(grid_c - tc)
                min_dist = np.minimum(min_dist, dist)
            
            # 將距離轉換為 0~1 的梯度。
            # 目標所在位置距離為 0，值為 1.0；距離越遠，值越小。
            max_dist = float(self.rows + self.cols)
            gradient_map = 1.0 - (min_dist / max_dist)
            gradient_map = np.clip(gradient_map, 0.0, 1.0) # 確保在安全範圍
        else:
            gradient_map = self.current_target_mask.astype(np.float32)

        # ==========================================
        # [NEW] 5. 未來腳位地圖 (Future Pins Map)
        # ==========================================
        future_map = np.zeros((self.rows, self.cols), dtype=np.float32)
        
        # 取得所有「還沒布線」的 Pin (index >= current_net_id)
        # 注意: pins 列表是按照 ID 排序的，current_net_id 對應的是當前正在布的 ID
        # 如果 current_net_id=1 (P0)，那麼 P1, P2... 都是未來
        # self.pins 的 index 與 net_id 對應關係需一致
        
        # 這裡假設 self.current_net_id 是當前正在布的 Net ID (1-based)
        # self.pins list index 是 0-based. 
        # 所以未來的 pin 是從 index = self.current_net_id 開始
        
        start_future_idx = self.current_net_id 
        if start_future_idx < len(self.pins):
            for i in range(start_future_idx, len(self.pins)):
                # 取得未來 Pin 的物理座標
                p_data = self.pins[i]
                fr, fc = p_data[1], p_data[2]
                
                # 將座標轉為整數格子
                r_idx, c_idx = int(fr), int(fc)
                
                # 在地圖上標記
                if 0 <= r_idx < self.rows and 0 <= c_idx < self.cols:
                    future_map[r_idx, c_idx] = 1.0 
                    # 你也可以考慮把未來 Pin 周圍的一圈設為 0.5 (警戒區)
                    # 但目前先標記本體就好

        # ==========================================
        # 堆疊 6 層 Input (原本是 5 層)
        # ==========================================
        return np.stack([obs_map, pos_map, v_lock, h_lock, gradient_map, future_map], axis=0)

    # [MODIFIED] 加入方向性與車道偏好邏輯
    # [MODIFIED] 加入 next_lane 參數以處理轉彎邏輯
    # [MODIFIED] 新增 last_action 參數以判斷轉彎方向 (Right Turn vs Left Turn)
    # [MODIFIED] 完整修正版：包含 Step 0 推導與詳細轉彎邏輯
    def _get_candidate_ring_paths(self, start_idx, end_idx, dr=0, dc=0, curr_lane=-1, next_lane=-1, last_action=None):
        if start_idx == end_idx: 
            return [[start_idx]]

        # [關鍵] 呼叫全域的 BFS 函式，它會根據 TILE_GRAPH 找到包含 12-15 在內的最短路徑
        shortest_path = get_shortest_ring_path(start_idx, end_idx)
        
        if not shortest_path:
            return []
        
        # 回傳包含 16 點結構中所有中間節點的路徑序列
        return [shortest_path]
        
    

    

    def step(self, action):
        dr, dc = self.actions[action]
        cr, cc = self.head_pos
        nr, nc = cr + dr, cc + dc
        
        # ---------------------------------------------------
        # 0. 基本檢查 (邊界、回頭路)
        # ---------------------------------------------------
        if (cr, cc, action) in self.blocked_moves:
            return self.get_state(), -20, True, False

        current_entry_occupier = self.node_occupancy[cr, cc, self.current_entry_node_idx]
        if current_entry_occupier != 0 and current_entry_occupier != self.current_net_id:
             return self.get_state(), -20, True, False

        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            return self.get_state(), -10, True, False
        
        if self.capacity_map[nr, nc] <= 0 or (nr, nc) in self.current_path:
             return self.get_state(), -10, True, False

        # ---------------------------------------------------
        # 1. 計算基礎獎勵
        # ---------------------------------------------------
        step_reward = -0.4
        
        if self.last_action is not None and action != self.last_action:
            step_reward -= 1.0
            
        opposite_action_map = {0: 1, 1: 0, 2: 3, 3: 2}
        if opposite_action_map[action] in self.path_action_history:
            step_reward -= 1.0 
        self.path_action_history.add(action)
        
        step_reward += self._calculate_lookahead_reward(self.head_pos, (nr, nc))
        
        if self.current_candidate_indices:
            tr, tc = (-1, -1)
            if self.primary_target_pos is not None: tr, tc = self.primary_target_pos
            is_secondary_rewarded = False
            for idx in self.current_candidate_indices:
                cand_r, cand_c = self.perimeter_list[idx]
                is_aligned = (nr == cand_r) or (nc == cand_c)
                if not is_aligned: continue
                prev_dist = abs(cr - cand_r) + abs(cc - cand_c)
                curr_dist = abs(nr - cand_r) + abs(nc - cand_c)
                if curr_dist < prev_dist:
                    if (cand_r, cand_c) == (tr, tc): step_reward += 1.0
                    elif not is_secondary_rewarded:
                         step_reward += 0.5; is_secondary_rewarded = True

        # ---------------------------------------------------
        # 2. 決定候選車道
        # ---------------------------------------------------
        preferred_exit_node = -1
        lane_reward = 0.0
        candidate_lanes = []
        
        if action == 0: candidate_lanes = [3, 2]
        elif action == 1: candidate_lanes = [2, 3]
        elif action == 2: candidate_lanes = [0, 1]
        elif action == 3: candidate_lanes = [1, 0]

        is_shared = (self.capacity_map[nr, nc] < self.initial_capacity) and (self.capacity_map[nr, nc] > 0)
        is_full_empty = (self.capacity_map[nr, nc] == self.initial_capacity)

        if self.primary_target_pos is not None:
            tr, tc = self.primary_target_pos
            if tr == self.rows - 1 and action == 1:
                if is_shared: preferred_exit_node = 4; lane_reward = 2.0
                elif is_full_empty: preferred_exit_node = 5; lane_reward = 0.1
            elif tr == 0 and action == 0:
                if is_shared: preferred_exit_node = 0; lane_reward = 2.0
                elif is_full_empty: preferred_exit_node = 1; lane_reward = 0.1
            elif tc == 0 and action == 2:
                if is_shared: preferred_exit_node = 6; lane_reward = 2.0
                elif is_full_empty: preferred_exit_node = 7; lane_reward = 0.1
            elif tc == self.cols - 1 and action == 3:
                if is_shared: preferred_exit_node = 2; lane_reward = 2.0
                elif is_full_empty: preferred_exit_node = 3; lane_reward = 0.1

        # ---------------------------------------------------
        # 3. 物理路徑搜尋與衝突解決
        # ---------------------------------------------------
        selected_lane = -1
        final_ring_path = None
        final_next_entry = -1
        
        sorted_lanes = []
        if preferred_exit_node != -1:
            first, second = [], []
            for ln in candidate_lanes:
                dummy_exit = self._get_exit_node_index(cr, cc, nr, nc, self.my_lane_idx, ln)
                if dummy_exit == preferred_exit_node: first.append(ln)
                else: second.append(ln)
            sorted_lanes = first + second
        else:
            sorted_lanes = candidate_lanes

        for target_lane in sorted_lanes:
            entry_node_idx = self.current_entry_node_idx
            exit_node_idx = self._get_exit_node_index(cr, cc, nr, nc, self.my_lane_idx, target_lane)
            
            candidate_paths = self._get_candidate_ring_paths(
                entry_node_idx, exit_node_idx, dr=dr, dc=dc, 
                curr_lane=self.my_lane_idx, next_lane=target_lane, last_action=self.last_action 
            )
            
            valid_path_found = None
            
            for path_nodes in candidate_paths:
                is_path_clear = True
                blocking_net_id = -1
                for n_idx in path_nodes:
                    occupier = self.node_occupancy[cr, cc, n_idx]
                    if occupier != 0 and occupier != self.current_net_id:
                        is_path_clear = False
                        blocking_net_id = occupier
                        break 
                
                if is_path_clear:
                    valid_path_found = path_nodes
                    break 
                else:
                    # --- 內部節點重新佈線 ---
                    if blocking_net_id > 0:
                        reroute_path = self.try_resolve_conflict_in_tile(
                            cr, cc, self.current_net_id, blocking_net_id, 
                            path_nodes[0], path_nodes[-1]
                        )
                        if reroute_path:
                            valid_path_found = reroute_path
                            # [修改] 重新佈線成功，給予獎勵
                            step_reward += 0.5 
                            break 

            # --- 下一格入口檢查 (整合入口協商) ---
            if valid_path_found is not None:
                next_entry_idx = self._get_entry_node_index(exit_node_idx)
                occupier_next = self.node_occupancy[nr, nc, next_entry_idx]
                
                if occupier_next != 0 and occupier_next != self.current_net_id:
                    net_a_id = occupier_next
                    is_resolved = False
                    seg_info = self._get_net_segment_entry_exit(net_a_id, nr, nc)
                    
                    if seg_info:
                        a_entry, a_exit = seg_info
                        if next_entry_idx == a_entry or next_entry_idx == a_exit:
                            valid_path_found = None # 情況 1: 硬阻擋
                        else:
                            # 情況 2: 入口處軟阻擋，嘗試叫別人讓開
                            reroute_res = self.try_resolve_conflict_in_tile(
                                nr, nc, self.current_net_id, net_a_id,
                                next_entry_idx, next_entry_idx
                            )
                            if reroute_res:
                                is_resolved = True
                                # [修改] 重新佈線成功，給予獎勵
                                step_reward += 0.5 
                            else:
                                valid_path_found = None
                    else:
                        valid_path_found = None
                    
                    if not is_resolved and valid_path_found is not None:
                        valid_path_found = None
            
            if valid_path_found is not None:
                selected_lane = target_lane
                final_ring_path = valid_path_found
                final_next_entry = next_entry_idx
                if preferred_exit_node != -1 and exit_node_idx == preferred_exit_node:
                    step_reward += lane_reward
                break 

        if selected_lane == -1:
            return self.get_state(), -20, True, False

        # ---------------------------------------------------
        # 4. 更新狀態
        # ---------------------------------------------------
        for n_idx in final_ring_path:
            self.node_occupancy[cr, cc, n_idx] = self.current_net_id
        
        self.head_pos = (nr, nc)
        self.current_path.append((nr, nc))
        self.current_path_entries.append(final_next_entry)
        self.my_lane_idx = selected_lane
        self.current_path_lanes.append(selected_lane)
        self.current_entry_node_idx = final_next_entry
        self.current_node_sequences.append(final_ring_path) 
        self.last_action = action

        # ---------------------------------------------------
        # 5. 結束判斷
        # ---------------------------------------------------
        if self.current_target_mask[nr, nc]:
            self._update_global_maps()
            self.node_occupancy[nr, nc, self.current_entry_node_idx] = self.current_net_id
            self.commited_paths[self.current_net_id] = (
                self.current_path.copy(), self.current_path_lanes.copy(), 
                self.escape_trace.copy(), self.current_node_sequences.copy()
            )
            success_reward = 150.0 if is_shared else 100.0
            return self.get_state(), success_reward, True, True

        if (nr == 0 or nr == self.rows-1 or nc == 0 or nc == self.cols-1):
            return self.get_state(), -20, True, False
        
        return self.get_state(), step_reward, False, False
    
    def _update_global_maps(self):
        if len(self.current_path) < 1: return
        visited_tiles = set()
        prev_exit_idx = -1

        for i in range(len(self.current_path)):
            r, c = self.current_path[i]
            lane_used = self.current_path_lanes[i]
            
            # 1. 決定 Entry Node
            if i == 0:
                # 優先使用紀錄的 Entry，這在 Direct Escape 非常重要
                if hasattr(self, 'current_path_entries') and i < len(self.current_path_entries):
                    entry_node = self.current_path_entries[i]
                else:
                    entry_node = 0 # Fallback
            else:
                entry_node = self._get_entry_node_index(prev_exit_idx)
            
            # 2. 決定 Path Nodes (這是最關鍵的一步)
            nodes_to_mark = []
            
            # [修正]：優先使用 current_node_sequences，這包含了協商後的結果
            if i < len(self.current_node_sequences) and len(self.current_node_sequences[i]) > 0:
                nodes_to_mark = self.current_node_sequences[i]
            else:
                # 若無紀錄，則根據 Entry/Exit 自動計算最短路徑
                exit_node = -1
                if i == len(self.current_path) - 1:
                    exit_node = get_exit_node_index_boundary(r, c, self.rows, self.cols, lane_used)
                else:
                    nr, nc = self.current_path[i+1]
                    nl = self.current_path_lanes[i+1]
                    exit_node = self._get_exit_node_index(r, c, nr, nc, lane_used, nl)
                
                nodes_to_mark = get_shortest_ring_path(entry_node, exit_node)

                # [重要] 將計算結果回填，確保資料一致性
                if i >= len(self.current_node_sequences):
                    self.current_node_sequences.append(nodes_to_mark)
                else:
                    self.current_node_sequences[i] = nodes_to_mark

            # 3. 更新佔用圖 (Occupancy Map)
            for n_idx in nodes_to_mark:
                # 強制寫入，不做檢查，確保地圖反映當前路徑
                self.node_occupancy[r, c, n_idx] = self.current_net_id
            
            # 更新 Exit 供下一格使用
            if nodes_to_mark:
                prev_exit_idx = nodes_to_mark[-1]

            # 4. 更新容量 (Capacity)
            if (r, c) not in visited_tiles:
                self.capacity_map[r, c] = max(0, self.capacity_map[r, c] - 1)
                visited_tiles.add((r, c))

            # 5. 更新方向鎖 (Direction Map)
            m_type = 0
            if i < len(self.current_path) - 1:
                nr, nc = self.current_path[i+1]
                m_type = 1 if r != nr else 2 
            elif i > 0:
                pr, pc = self.current_path[i-1]
                m_type = 1 if r != pr else 2
            
            if self.direction_map[r, c] == 0: self.direction_map[r, c] = m_type
            elif self.direction_map[r, c] != m_type: self.direction_map[r, c] = 3

    def _update_net_history(self, net_id, r, c, new_node_seq):
        """
        當發生重佈線時，更新舊 Net 的歷史紀錄。
        """
        if net_id not in self.commited_paths:
            return

        # 取出該 Net 的歷史資料
        # tuple 是不可變的，所以我們取出後要轉成 list 修改，最後再存回 tuple
        data = self.commited_paths[net_id]
        if len(data) == 4:
            path_coords, path_lanes, esc_trace, node_seqs = data
        else:
            # 相容舊格式
            path_coords, path_lanes, esc_trace = data
            node_seqs = [[] for _ in range(len(path_coords))]

        # 找到 (r, c) 在該 Net 路徑中的 index
        target_idx = -1
        for i, (pr, pc) in enumerate(path_coords):
            if pr == r and pc == c:
                target_idx = i
                break
        
        # 如果找到了，就更新那一格的內部節點序列
        if target_idx != -1:
            # 確保 node_seqs 足夠長 (防止 index error)
            while len(node_seqs) <= target_idx:
                node_seqs.append([])
            
            # [DEBUG]
            # print(f"  [Reroute Log] Net {net_id} at ({r},{c}): {node_seqs[target_idx]} -> {new_node_seq}")

            # 替換成新的路徑
            node_seqs[target_idx] = new_node_seq
            
            # 更新回資料庫
            self.commited_paths[net_id] = (path_coords, path_lanes, esc_trace, node_seqs)
# ==========================================
# 3. 視覺化繪圖 & 輔助 (20x20 Label)
# ==========================================




def get_entry_node_index(prev_exit_idx):
    return { 4: 1, 5: 0, 0: 5, 1: 4, 2: 7, 3: 6, 6: 3, 7: 2 }.get(prev_exit_idx, 0)

def get_exit_node_index_by_direction(env, curr_r, curr_c, next_r, next_c, curr_lane, next_lane, net_id):
    """同步環境內部的 _get_exit_node_index 邏輯"""
    dr, dc = next_r - curr_r, next_c - curr_c
    
    if dr == -1: # Up
        return 1 if next_lane == 3 else 0
    elif dr == 1: # Down
        return 5 if next_lane == 2 else 4
    elif dc == -1: # Left
        return 7 if next_lane == 0 else 6
    elif dc == 1: # Right
        return 3 if next_lane == 1 else 2
        
    return 0

def get_exit_node_index_boundary(r, c, rows, cols, lane):
    is_bottom = (r == rows - 1)
    is_top = (r == 0)
    is_right = (c == cols - 1)
    is_left = (c == 0)
    if is_bottom: return 4 if lane == 3 else 5
    if is_top:    return 1 if lane == 3 else 0
    if is_right:  return 3 if lane == 1 else 2
    if is_left:   return 6 if lane == 1 else 7
    return 0







# [修正版] plot_pcb_visual_style
# 修正內容：
# 1. 確保畫線能連到最後的 Pad 中心
# 2. 修正 Direct Escape 顏色與線寬

def plot_pcb_visual_style(env, ax, current_pin_idx, finished_paths, current_reward, status_text="", target_mask=None):
    ax.clear()
    ax.set_title(f"20x20 High Density (16-Node Complex Tile) | Reward: {current_reward:.1f} | {status_text}", fontsize=11)
    
    view_pad = 1.0
    ax.set_xlim(-view_pad, env.cols + view_pad - 1)
    ax.set_ylim(-view_pad, env.rows + view_pad - 1)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('white')
    
    boundary_centers = {} 
    pad_size = 0.6
    
    def draw_boundary_pad(x, y, text, b_idx):
        rect = patches.Rectangle((x - pad_size/2, y - pad_size/2), pad_size, pad_size, 
                                 facecolor='cyan', edgecolor='black', zorder=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=3.5, zorder=3)
        boundary_centers[b_idx] = (y, x) 

    # 1. 繪製邊界 Pad (保持不變)
    for c in range(env.cols): draw_boundary_pad(c, env.rows - 0.2, f"B{c}", c)
    for r in range(env.rows - 1, -1, -1): draw_boundary_pad(env.cols - 0.2, r, f"B{21 + (env.rows - 1 - r)}", 21 + (env.rows - 1 - r))
    for c in range(env.cols - 1, -1, -1): draw_boundary_pad(c, -0.8, f"B{42 + (env.cols - 1 - c)}", 42 + (env.cols - 1 - c))
    for r in range(env.rows): draw_boundary_pad(-0.8, r, f"B{63 + r}", 63 + r)

    # =================================================================
    # [新增部分] 繪製每個 Tile 的內部 16 點結構底圖
    # =================================================================
    # 定義要畫的底圖連線 (根據 TILE_GRAPH)
    background_lines = [
        # 外圈區段
        (0, 8), (8, 1), (1, 2), (2, 9), (9, 3), (3, 4),
        (4, 10), (10, 5), (5, 6), (6, 11), (11, 7), (7, 0),
        # 輻條 (Spokes)
        (8, 12), (9, 13), (10, 14), (11, 15),
        # 內圈 (Inner Diamond)
        (12, 13), (13, 14), (14, 15), (15, 12)
    ]
    
    for r in range(env.rows):
        for c in range(env.cols):
            # 取得該格子的 16 個物理座標
            nodes_phys = get_octagon_nodes(r, c)
            # 畫出定義好的連線，使用淺灰色
            for u, v in background_lines:
                p1 = nodes_phys[u]
                p2 = nodes_phys[v]
                # zorder 設為 5，比背景高，但比實際路徑低
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#D3D3D3', lw=0.6, zorder=5, alpha=0.6)
            
            # (選用) 畫出小小的節點點位，幫助除錯看位置
            # px, py = zip(*nodes_phys)
            # ax.scatter(px, py, s=2, color='#C0C0C0', zorder=6, alpha=0.6)
            
            # (選用) 標示內圈節點編號 (除錯用)
            # ax.text(nodes_phys[12][0], nodes_phys[12][1], "12", fontsize=2, ha='center', color='gray', zorder=7)

    # =================================================================

    # 2. 繪製背景障礙點 (保持不變)
    pin_radius = 0.22 
    active_pin_coords = [(p[1], p[2]) for p in env.pins]
    for r in range(env.rows-1):
        for c in range(env.cols-1):
            cand_pr, cand_pc = r + 0.5, c + 0.5
            is_active = any(abs(cand_pr - apr) < 0.1 and abs(cand_pc - apc) < 0.1 for (apr, apc) in active_pin_coords)
            if not is_active:
                circle = patches.Circle((cand_pc, cand_pr), pin_radius, facecolor='black', 
                                     edgecolor='black', lw=0.5, zorder=1)
                ax.add_patch(circle)
     
    # 3. 繪製主動腳位 Pin (保持不變)
    for idx, p_data in enumerate(env.pins):
        pid, pr_phys, pc_phys = p_data[:3]
        circle = patches.Circle((pc_phys, pr_phys), pin_radius, facecolor='limegreen', edgecolor='black', zorder=15)
        ax.add_patch(circle)
        ax.text(pc_phys, pr_phys, f"P{pid}", ha='center', va='center', fontsize=4, fontweight='bold', zorder=16)

    PATH_COLORS = ['#0000CD', '#FF8C00', '#006400', '#008B8B', '#8B4513', '#4682B4', '#556B2F', '#191970', '#8A2BE2', '#A52A2A', '#5F9EA0']
    ROUTE_WIDTH = 2.0 
     
    # 4. 繪製線條 (邏輯基本不變，自動適應新的節點數量)
    for pid, data in finished_paths.items():
        actual_node_seqs = []
        if len(data) == 4:
            path_coords, path_lanes, esc_trace, actual_node_seqs = data
        else:
            path_coords, path_lanes, esc_trace = data

        if len(path_coords) < 1: continue
        path_color = PATH_COLORS[pid % len(PATH_COLORS)]
        
        # Direct Escape
        if path_lanes[0] == -1: 
            ax.plot([esc_trace[0][1], esc_trace[2][1]], [esc_trace[0][0], esc_trace[2][0]], color=path_color, lw=ROUTE_WIDTH, zorder=10)
            continue

        full_draw_points = []
        # 加入起點 Trace (Pin -> Via -> Node)
        if esc_trace:
            full_draw_points.append((esc_trace[0][1], esc_trace[0][0]))
            if len(esc_trace) > 1: full_draw_points.append((esc_trace[1][1], esc_trace[1][0]))
            if len(esc_trace) > 2: full_draw_points.append((esc_trace[2][1], esc_trace[2][0]))

        prev_exit_idx = -1
        
        for i in range(len(path_coords)):
            curr_r, curr_c = path_coords[i]
            curr_lane = path_lanes[i]
            
            # 取得該格子的 16 個物理座標
            tile_phys = get_octagon_nodes(curr_r, curr_c)

            # 使用實際紀錄的路徑序列 (這裡的 index 現在範圍是 0-15)
            if actual_node_seqs and i < len(actual_node_seqs):
                ring_nodes = actual_node_seqs[i]
                # 將序列中的 index 轉換為物理座標
                for ridx in ring_nodes: 
                    full_draw_points.append(tile_phys[ridx])
                if ring_nodes:
                    prev_exit_idx = ring_nodes[-1]
            else:
                # Fallback for older data structures or calculated paths
                # (這部分在使用新的環境代碼時應該較少觸發，但保持兼容性)
                if i == len(path_coords) - 1: 
                     # 注意：這裡的 get_exit_node_index_boundary 仍回傳舊的 N0-N7，
                     # 但因為 N0-N7 在新結構中座標沒變，所以畫圖是正確的。
                    exit_idx = get_exit_node_index_boundary(curr_r, curr_c, env.rows, env.cols, curr_lane)
                    full_draw_points.append(tile_phys[exit_idx])

        # 連接到最終 Pad
        if full_draw_points:
            last_pt = full_draw_points[-1]
            min_d = 1.5; nearest = None
            for b_idx, (pr, pc) in boundary_centers.items():
                d = math.sqrt((last_pt[0]-pc)**2 + (last_pt[1]-pr)**2)
                if d < min_d: min_d = d; nearest = (pc, pr)
            if nearest:
                fx, fy = nearest; last_l = path_lanes[-1]; OFF = 0.12
                # 這裡的 Lane offset 邏輯可能需要根據新的節點位置微調，但暫時維持原樣
                if last_l == 2: fx -= OFF
                elif last_l == 3: fx += OFF
                elif last_l == 0: fy -= OFF
                elif last_l == 1: fy += OFF
                full_draw_points.append((fx, fy))

        px, py = zip(*full_draw_points)
        ax.plot(px, py, color=path_color, lw=ROUTE_WIDTH, solid_capstyle='round', zorder=10)

    plt.draw(); plt.pause(0.001)
# ==========================================
# 4. 環狀邏輯輔助函式
# ==========================================
def generate_perimeter_path(rows, cols):
    perimeter = []
    # Bottom: Row 20, C 0->20
    for c in range(cols): perimeter.append((rows - 1, c)) 
    # Right: Col 20, R 20->0
    for r in range(rows - 1, -1, -1): perimeter.append((r, cols - 1))
    # Top: Row 0, C 20->0
    for c in range(cols - 1, -1, -1): perimeter.append((0, c))
    # Left: Col 0, R 0->20
    for r in range(rows): perimeter.append((r, 0))
    return perimeter




def get_advanced_cursor(r, c, lane, current_idx, rows, cols, total_len):
    step = 0
    if r == rows - 1: # Bottom
        if lane in [0, 3]: step = 1
    elif c == cols - 1: # Right
        if lane in [0, 2]: step = 1
    elif r == 0: # Top
        if lane in [1, 2]: step = 1
    elif c == 0: # Left
        if lane in [1, 3]: step = 1
    return (current_idx + step) % total_len

def plot_training_curves(history):
    epochs = range(len(history['rewards']))
    plt.style.use('ggplot')
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Metrics', fontsize=16)
    axs[0, 0].plot(epochs, history['rewards']); axs[0, 0].set_title('Reward')
    axs[0, 1].plot(epochs, history['losses']); axs[0, 1].set_title('Loss')
    axs[1, 0].plot(epochs, history['success_rates']); axs[1, 0].set_title('Success Rate')
    axs[1, 1].plot(epochs, history['avg_steps']); axs[1, 1].set_title('Steps')
    plt.tight_layout(); plt.show()

def print_detailed_statistics(env, finished_paths):
    print("\n" + "="*85)
    print(f"{'FINAL ROUTING STATISTICS (Detailed Node Trace)':^85}")
    print("="*85)
    sorted_pids = sorted(finished_paths.keys())
    for pid in sorted_pids:
        # 解包檢查
        data = finished_paths[pid]
        actual_node_seqs = []
        if len(data) == 4:
            path_coords, path_lanes, escape_trace, actual_node_seqs = data
        else:
            path_coords, path_lanes, escape_trace = data

        print(f"\n[Pin P{pid}] Total Length: {len(path_coords)} tiles")
        print(f"{'Step':<5} | {'Tile (R, C)':<12} | {'Lane':<5} | {'Path Nodes (Sequence)':<40}")
        print("-" * 85)
        
        prev_exit_idx = -1
        for i in range(len(path_coords)):
            curr_r, curr_c = path_coords[i]
            curr_lane = path_lanes[i]
            
            # 準備顯示的 Node 序列
            nodes_str = ""
            if actual_node_seqs and i < len(actual_node_seqs):
                # 這是實際走過的路徑
                nodes_str = str(actual_node_seqs[i])
            else:
                # 這是最後一格或是 Direct Escape，顯示為 (Calculated)
                # 我們重新計算一次來顯示理論值
                if i == 0: 
                    if escape_trace and len(escape_trace) >= 3:
                        entry_idx = env._get_closest_node_index(curr_r, curr_c, escape_trace[2][1], escape_trace[2][0])
                    else: entry_idx = 0
                else: 
                    entry_idx = get_entry_node_index(prev_exit_idx)

                if i == len(path_coords) - 1:
                    exit_idx = get_exit_node_index_boundary(curr_r, curr_c, env.rows, env.cols, curr_lane)
                else:
                    nr, nc = path_coords[i+1]; nl = path_lanes[i+1]
                    exit_idx = get_exit_node_index_by_direction(env, curr_r, curr_c, nr, nc, curr_lane, nl, pid+1)
                
                # 若無紀錄，顯示計算出的最短路徑
                calc_path = get_shortest_ring_path(entry_idx, exit_idx)
                nodes_str = f"{calc_path} (Auto)"

            print(f"{i:<5} | ({curr_r:2d}, {curr_c:2d})   | {curr_lane:<5} | {nodes_str:<40}")
            
            # 更新 prev_exit 供下一次計算理論值用 (即使這次是用 actual)
            # 這裡我們需要知道這次的 exit 是什麼。如果有 actual，取最後一個；沒有則算出來。
            if actual_node_seqs and i < len(actual_node_seqs):
                prev_exit_idx = actual_node_seqs[i][-1]
            else:
                # Fallback calculation
                if i < len(path_coords) - 1:
                     nr, nc = path_coords[i+1]; nl = path_lanes[i+1]
                     prev_exit_idx = get_exit_node_index_by_direction(env, curr_r, curr_c, nr, nc, curr_lane, nl, pid+1)

    print("="*85 + "\n")
# [MODIFIED] Added cursor logic for DIRECT escape
# [MODIFIED] Added cursor logic for DIRECT escape and is_direct_done check
# [MODIFIED] Updated for D3QN (Dueling Double DQN)
# [MODIFIED] 整合了新的單側軟性牆壁邏輯與重置機制
def run_integrated_demo():
    # ==========================================
    # [GPU] 0. 設定裝置
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")
    if device.type == 'cuda':
        print(f"GPU 名稱: {torch.cuda.get_device_name(0)}")

    # ==========================================
    # 1. 設定與初始化
    # ==========================================
    cols = 21 
    rows = 21
    capacity = 4 
    
    # --- [設定] 訓練模式 ---
    DO_TRAIN = True
    SHOW_DETAILED_STATS = True  # 控制是否印出詳細節點追蹤
    
    MODEL_PATH = "pcb_d3qn_20x20_tilecap4.pth"
    BEST_MODEL_PATH = "pcb_d3qn_best_20x20.pth" # 最佳模型的檔名

    # --- [4090 優化參數] ---
    BATCH_SIZE = 256          # 加大 Batch Size，榨乾硬體效能
    UPDATE_EVERY = 4          # 每 4 步才更新一次網路 (穩定梯度並加速)
    MEMORY_CAPACITY = 200000  # 加大記憶體
    SAVE_FREQ = 100           # 每 100 回合自動存檔

    # 完整保留原有的 42 個座標
    pins = [
        (0, 16.5, 5.5, 0), (1, 15.5, 3.5, 1), (2, 11.5, 2.5, 2), (3, 13.5, 8.5, 3),
        (4, 15.5, 10.5, 4), (5, 13.5, 13.5, 5), (6, 16.5, 12.5, 6), (7, 12.5, 13.5, 7),
        (8, 14.5, 15.5, 8), (9, 14.5, 17.5, 9), (10, 17.5, 17.5, 10), (11, 19.5, 18.5, 11),
        (12, 17.5, 19.5, 12), (13, 11.5, 17.5, 13), (14, 12.5, 19.5, 14), (15, 11.5, 18.5, 15),
        (16, 9.5, 12.5, 16), (17, 7.5, 19.5, 17), (18, 8.5, 14.5, 18), (19, 4.5, 18.5, 19),
        (20, 3.5, 15.5, 20), (21, 0.5, 19.5, 21), (22, 1.5, 17.5, 22), (23, 2.5, 14.5, 23),
        (24, 7.5, 11.5, 24), (25, 4.5, 11.5, 25), (26, 5.5, 9.5, 26), (27, 3.5, 8.5, 27),
        (28, 1.5, 6.5, 28), (29, 0.5, 4.5, 29), (30, 0.5, 3.5, 30), (31, 0.5, 5.5, 31),
        (32, 2.5, 1.5, 32), (33, 2.5, 6.5, 33), (34, 4.5, 1.5, 34), (35, 8.5, 8.5, 35),
        (36, 7.5, 4.5, 36), (37, 9.5, 2.5, 37), (38, 12.5, 1.5, 38), (39, 14.5, 2.5, 39),
        (40, 17.5, 1.5, 40), (41, 19.5, 0.5, 41)
    ]
    
    env = PCBGridEnv(rows=rows, cols=cols, capacity=capacity, pins=pins)
    
    # 初始化 D3QN 模型
    dqn = PCBRouterD3QN(rows, cols, 4).to(device)
    dqn_target = PCBRouterD3QN(rows, cols, 4).to(device)
    dqn_target.load_state_dict(dqn.state_dict()) 
    dqn_target.eval() 
    
    optimizer = optim.Adam(dqn.parameters(), lr=0.0001)
    memory = deque(maxlen=MEMORY_CAPACITY) # 套用加大的 Replay Buffer
    perimeter_list = generate_perimeter_path(rows, cols)
    perimeter_len = len(perimeter_list)
    boundary_cap_per_tile = 2 

    # ==========================================
    # 2. 訓練階段
    # ==========================================
    training_time_total = 0.0 
    target_update_freq = 200 
    total_updates = 0 

    if DO_TRAIN:
        training_episodes = 5000  # 可以根據需求調高至 10000
        epsilon_start = 0.9          
        epsilon_end = 0.05          
        epsilon_decay = 1000        
        history = { 'rewards': [], 'losses': [], 'success_rates': [], 'avg_steps': [] }
        
        print(f"開始訓練 D3QN [Optimized] (Grid: {rows}x{cols}, Batch: {BATCH_SIZE})...")
        train_start_time = time.time() 
        
        # =========== [新增] 初始化最佳分數 ===========
        best_reward = -float('inf')
        # ===========================================

        for ep in range(training_episodes):
            current_eps = epsilon_end + (epsilon_start - epsilon_end) * \
                          math.exp(-1. * ep / epsilon_decay)

            if ep % 50 == 0: 
                print(f"Training Episode: {ep} | Eps: {current_eps:.3f}")

            # 重置環境狀態
            env.capacity_map.fill(capacity)
            env.direction_map.fill(0)
            env.sub_lane_map.fill(False)
            env.node_occupancy.fill(0)
            env.boundary_via_map = {} 
            env.blocked_moves.clear()
            
            train_cursor = None 
            train_usage = {i: 0 for i in range(perimeter_len)}
            ep_r = 0; ep_loss = 0; ep_lc = 0; ep_succ = 0; ep_steps = 0; p_att = 0
            
            for pin_idx in range(len(pins)):
                p_r, p_c = pins[pin_idx][1], pins[pin_idx][2]
                
                if train_cursor is None:
                    train_cursor = get_closest_perimeter_index(p_r, p_c, perimeter_list, rows, cols)
                
                valid_indices = get_best_candidate_indices_circular(
                    p_r, p_c, perimeter_list, train_cursor, train_usage, boundary_cap_per_tile, rows, cols
                )
                
                if not valid_indices: break 
                
                target_mask = np.zeros((rows, cols), dtype=bool)
                allowed_count = 0
                
                for idx in valid_indices[:3]: 
                    if allowed_count >= 3: break
                    r, c = perimeter_list[idx]
                    target_mask[r, c] = True
                    allowed_count += 1
                
                tr_tile, tc_tile = perimeter_list[valid_indices[0]]
                target_phys_pos = get_boundary_phys_coords(tr_tile, tc_tile, rows, cols)

                env.prepare_pin_routing(target_mask, pin_idx + 1, valid_indices[:3])
                
                curr_c_val = train_cursor if train_cursor is not None else 0
                possible_starts = env.get_possible_starts(pin_idx, target_phys_pos, current_cursor=curr_c_val)
                
                if not possible_starts: continue
                
                p_att += 1
                state = env.set_start_tile(possible_starts[0], (p_r, p_c))
                if state is None: continue

                if env.is_direct_done:
                    done = True; success = True; steps = 0
                else:
                    done = False; steps = 0; success = False
                
                while not done and steps < 100:
                    if random.random() < current_eps: 
                        action = random.randint(0, 3)
                    else:
                        with torch.no_grad(): 
                            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                            action = dqn(state_t).argmax().item()
                    
                    next_state, reward, done, success = env.step(action)
                    ep_r += reward
                    memory.append((state, action, reward, next_state, done))
                    state = next_state; steps += 1
                    
                    # ====================================================
                    # [優化核心] 配合 UPDATE_EVERY 降低更新頻率，加大 Batch
                    # ====================================================
                    if len(memory) > BATCH_SIZE and steps % UPDATE_EVERY == 0: 
                        batch = random.sample(memory, BATCH_SIZE)
                        bs, ba, br, bns, bd = zip(*batch)
                        
                        bs_t = torch.FloatTensor(np.array(bs)).to(device)
                        ba_t = torch.LongTensor(ba).unsqueeze(1).to(device)
                        br_t = torch.FloatTensor(br).unsqueeze(1).to(device)
                        bns_t = torch.FloatTensor(np.array(bns)).to(device)
                        bd_t = torch.FloatTensor(bd).unsqueeze(1).to(device)
                        
                        q = dqn(bs_t).gather(1, ba_t)
                        next_actions = dqn(bns_t).argmax(dim=1, keepdim=True)
                        nq = dqn_target(bns_t).gather(1, next_actions)
                        
                        target_q = (br_t + 0.9 * nq * (1 - bd_t)).detach()
                        
                        # 改用 SmoothL1Loss (Huber Loss) 增強穩定性
                        loss = nn.SmoothL1Loss()(q, target_q)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        # 加入梯度裁剪
                        nn.utils.clip_grad_norm_(dqn.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                        ep_loss += loss.item()
                        ep_lc += 1
                        total_updates += 1

                        if total_updates % target_update_freq == 0:
                            dqn_target.load_state_dict(dqn.state_dict())
                
                ep_steps += steps
                if success:
                    ep_succ += 1
                    if env.is_direct_done:
                         direct_target_phys = env.escape_trace[-1]
                         best_p_idx = -1
                         for p_idx, (pr_p, pc_p) in enumerate(perimeter_list):
                            b_phys = get_boundary_phys_coords(pr_p, pc_p, rows, cols)
                            dist = abs(b_phys[0] - direct_target_phys[0]) + abs(b_phys[1] - direct_target_phys[1])
                            if dist < 0.6:
                                best_p_idx = p_idx; break
                         if best_p_idx != -1:
                             train_cursor = (best_p_idx + 1) % perimeter_len
                             train_usage[best_p_idx] += 1
                    else:
                        found_idx = -1
                        for idx in valid_indices:
                            if perimeter_list[idx] == env.current_path[-1]: found_idx = idx; break
                        
                        if found_idx != -1:
                            train_usage[found_idx] += 1
                            if train_usage[found_idx] < boundary_cap_per_tile:
                                train_cursor = found_idx 
                            else:
                                train_cursor = (found_idx + 1) % perimeter_len 

            history['rewards'].append(ep_r)
            history['losses'].append(ep_loss / ep_lc if ep_lc > 0 else 0)
            history['success_rates'].append(ep_succ / len(pins))
            history['avg_steps'].append(ep_steps / p_att if p_att > 0 else 0)

            # =========== [新增] 判斷是否為最佳模型 ===========
            if ep_r > best_reward:
                best_reward = ep_r
                torch.save(dqn.state_dict(), BEST_MODEL_PATH)
                print(f"🔥 發現新高分: {best_reward:.2f} | 最佳模型已保存至 {BEST_MODEL_PATH}")
            # ===============================================
            
            # [新增] 自動存檔機制
            if ep > 0 and ep % SAVE_FREQ == 0:
                torch.save(dqn.state_dict(), MODEL_PATH)
                print(f"--- [AutoSave] 模型已自動存檔於 Episode {ep} ---")

        train_end_time = time.time()
        training_time_total = train_end_time - train_start_time
        print(f"訓練完成！總耗時: {training_time_total:.2f} 秒")
        
        # 把訓練時間寫入 JSON 檔案
        try:
            with open("training_meta.json", "w") as f:
                json.dump({"training_time_seconds": training_time_total}, f)
        except Exception as e:
            print(f"儲存訓練時間失敗: {e}")
            
        torch.save(dqn.state_dict(), MODEL_PATH)
        print(f"最後模型已保存至: {MODEL_PATH}")
        
        plot_training_curves(history)

    else:
        # [修改] 優先讀取 BEST_MODEL_PATH，若無則讀取 MODEL_PATH
        target_load_path = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else MODEL_PATH
        if os.path.exists(target_load_path):
            print(f"正在讀取模型: {target_load_path} ...")
            dqn.load_state_dict(torch.load(target_load_path, map_location=device))
            dqn.eval()
        else:
            print(f"錯誤：找不到模型檔案 {target_load_path}，請先設定 DO_TRAIN = True 進行訓練。")
            return

    # ==========================================
    # 3. 推論/Demo 階段
    # ==========================================
    print("\n" + "="*30)
    print("Demo Start... (Inference Phase)")
    print(f"Total Perimeter Length: {perimeter_len}")
    fig, ax = plt.subplots(figsize=(12, 12))
    
    routing_start_time = time.time()
    wire_lengths = {} 

    for demo_round in range(1):
        # 重置環境狀態
        env.capacity_map.fill(capacity)
        env.direction_map.fill(0)
        env.sub_lane_map.fill(False)
        env.node_occupancy.fill(0)
        env.boundary_via_map = {} 
        env.blocked_moves.clear()
        env.commited_paths = {}
        finished_paths = {}; demo_usage = {i: 0 for i in range(perimeter_len)}
        demo_cursor = None
        
        for pin_idx in range(len(pins)):
            pid = pins[pin_idx][0]; p_r, p_c = pins[pin_idx][1], pins[pin_idx][2]
            
            if demo_cursor is None:
                demo_cursor = get_closest_perimeter_index(p_r, p_c, perimeter_list, rows, cols)
                start_label = get_boundary_name(perimeter_list[demo_cursor][0], perimeter_list[demo_cursor][1], rows, cols)
                print(f"--> [Init] Pin P{pid} determines start. Closest Boundary: {start_label} (Index {demo_cursor})")

            valid_indices = get_best_candidate_indices_circular(
                p_r, p_c, perimeter_list, demo_cursor, demo_usage, boundary_cap_per_tile, rows, cols
            )
            
            candidate_labels = []
            if valid_indices:
                first_idx = valid_indices[0]
                r_tgt, c_tgt = perimeter_list[first_idx]
                target_phys_pos = get_boundary_phys_coords(r_tgt, c_tgt, rows, cols)
                primary_target_label = get_boundary_name(r_tgt, c_tgt, rows, cols)
                
                for v_idx in valid_indices[:3]: 
                    vr, vc = perimeter_list[v_idx]
                    b_name = get_boundary_name(vr, vc, rows, cols)
                    candidate_labels.append(f"{b_name}({v_idx})")
            else:
                 primary_target_label = "None"
                 target_phys_pos = (p_r, p_c)

            cursor_label = get_boundary_name(perimeter_list[demo_cursor][0], perimeter_list[demo_cursor][1], rows, cols)
            
            print(f"Pin P{pid} | Current Cursor: {cursor_label} ({demo_cursor}) | Target Preference: {primary_target_label}")
            print(f"    -> Top 3 Candidates: {', '.join(candidate_labels)}")

            if not valid_indices: continue
                
            target_mask = np.zeros((rows, cols), dtype=bool); has_t = False
            for idx in valid_indices[:3]: 
                target_mask[perimeter_list[idx]] = True; has_t = True
            if not has_t: continue
            
            env.prepare_pin_routing(target_mask, pin_idx + 1, valid_indices[:3])
            
            curr_c_val = demo_cursor if demo_cursor is not None else 0
            possible_starts = env.get_possible_starts(pin_idx, target_phys_pos, current_cursor=curr_c_val)
            
            if not possible_starts: continue
            
            state = env.set_start_tile(possible_starts[0], (p_r, p_c))
            if state is None: 
                print(f"Pin P{pid} 起點 Node 已被佔用，跳過此候選起點")
                continue 
            
            if env.is_direct_done:
                done = True; succ = True; steps = 0
                direct_target_phys = env.escape_trace[-1] 
                matched_indices = []

                for p_idx, (pr_p, pc_p) in enumerate(perimeter_list):
                    b_phys = get_boundary_phys_coords(pr_p, pc_p, rows, cols)
                    dist = abs(b_phys[0] - direct_target_phys[0]) + abs(b_phys[1] - direct_target_phys[1])
                    if dist < 0.6: 
                        matched_indices.append(p_idx)
                
                if matched_indices:
                    prev_c = demo_cursor
                    if 0 in matched_indices and (perimeter_len - 1) in matched_indices:
                        demo_cursor = 0
                    else:
                        demo_cursor = max(matched_indices)
                    
                    landed_label = get_boundary_name(perimeter_list[demo_cursor][0], perimeter_list[demo_cursor][1], rows, cols)
                    print(f"    -> [Direct] Success! Landed near {matched_indices}. Cursor updated: {prev_c} -> {demo_cursor} ({landed_label})")
                else:
                    print(f"    -> [Direct] Warning: Could not find matching boundary index for pos {direct_target_phys}")

            else:
                done = False; succ = False; steps = 0
                
                while not done and steps < 100:
                    with torch.no_grad():
                        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                        action = dqn(state_t).argmax().item()
                    
                    next_state, r, done, succ = env.step(action)
                    state = next_state
                    steps += 1
            
            if succ:
                finished_paths[pid] = (
                    env.current_path.copy(), 
                    env.current_path_lanes.copy(), 
                    env.escape_trace.copy(), 
                    env.current_node_sequences.copy()
                )

                if not env.is_direct_done:
                    found_idx = -1
                    for idx in valid_indices:
                        if perimeter_list[idx] == env.current_path[-1]: found_idx = idx; break
                    
                    if found_idx != -1:
                        prev_c = demo_cursor
                        demo_usage[found_idx] += 1
                        landed_label = get_boundary_name(perimeter_list[found_idx][0], perimeter_list[found_idx][1], rows, cols)

                        force_advance = False
                        used_lane_idx = env.current_path_lanes[-1] 
                        target_tile_r, target_tile_c = perimeter_list[found_idx]
                        
                        if target_tile_r == rows - 1: 
                            if used_lane_idx == 3: force_advance = True
                        elif target_tile_c == cols - 1: 
                            if used_lane_idx == 0: force_advance = True
                        elif target_tile_r == 0: 
                            if used_lane_idx == 2: force_advance = True
                        elif target_tile_c == 0: 
                            if used_lane_idx == 1: force_advance = True

                        if demo_usage[found_idx] >= boundary_cap_per_tile or force_advance:
                            demo_cursor = (found_idx + 1) % perimeter_len
                            action_str = "Advance" + (" (Forced)" if force_advance else " (Full)")
                        else:
                            demo_cursor = found_idx
                            action_str = "Stay (Shared)"
                        
                        print(f"    -> Success! Landed at {landed_label}. Usage: {demo_usage[found_idx]}/{boundary_cap_per_tile} (Lane {used_lane_idx}). Cursor: {prev_c} -> {demo_cursor} ({action_str})")
        
        routing_end_time = time.time()
        routing_time_total = routing_end_time - routing_start_time
        
        print(">> Synchronizing final paths from Environment (handling reroutes)...")
        for pid in range(len(pins)):
            net_id = pid + 1
            if net_id in env.commited_paths:
                finished_paths[pid] = env.commited_paths[net_id]
                path_coords = env.commited_paths[net_id][0]
                path_lanes = env.commited_paths[net_id][1]
                
                # ==========================================
                # [核心修正] 對齊 MMCF 模型的線長計算邏輯
                # ==========================================
                if len(path_lanes) > 0 and path_lanes[0] == -1:
                    # Direct Escape (直接跳脫): 僅消耗 1 條外部邊界資源
                    calc_length = 1
                else:
                    # 一般跳脫路徑: 網格移動次數 + 最後連出邊界 = 剛好等於 len(path_coords)
                    calc_length = (len(path_coords) - 1) + 1
                    
                wire_lengths[pid] = calc_length

        print("\n" + "="*65)
        print(f"{'PERFORMANCE REPORT':^65}")
        print("="*65)
        
        # 讀取 JSON 顯示訓練時間
        if DO_TRAIN:
            print(f"Total Training Time : {training_time_total:.4f} seconds")
        else:
            if os.path.exists("training_meta.json"):
                try:
                    with open("training_meta.json", "r") as f:
                        meta = json.load(f)
                        saved_time = meta.get("training_time_seconds", 0.0)
                    print(f"Total Training Time : {saved_time:.4f} seconds (Loaded from log)")
                except:
                    print(f"Total Training Time : (Error reading log, loaded from pre-trained model)")
            else:
                print(f"Total Training Time : (Log missing, loaded from pre-trained model)")

        print(f"Total Routing Time  : {routing_time_total:.4f} seconds")
        print(f"Total Pins Routed   : {len(finished_paths)} / {len(pins)}")
        print("-" * 65)
        print(f"{'Pin ID':<10} | {'Wire Length (Tile Edges)':<20}")
        print("-" * 35)
        total_wire_len = 0
        for pid in sorted(wire_lengths.keys()):
            length = wire_lengths[pid]
            total_wire_len += length
            print(f"P{pid:<9} | {length:<20}")
        print("-" * 35)
        print(f"Total Wire Length   : {total_wire_len}")
        print("="*65 + "\n")

        if SHOW_DETAILED_STATS:
            print_detailed_statistics(env, finished_paths)
            
        plot_pcb_visual_style(env, ax, len(pins)-1, finished_paths, 0, f"20x20 Final Demo", target_mask=None)
        plt.show()

if __name__ == "__main__":
    run_integrated_demo()