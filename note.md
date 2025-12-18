"""
        功能：获取指定玩家的观测信息（深拷贝）
        
        输入参数：
            player (str, optional): 玩家标识，'A' 或 'B'
                若为 None，则返回当前击球方的观测
        
        返回值：
            tuple: (balls, my_targets, table)
            
                balls (dict): 球状态字典，{ball_id: Ball对象}
                    ball_id 取值：
                        - 'cue': 白球
                        - '1'-'7': 实心球（solid）
                        - '8': 黑8
                        - '9'-'15': 条纹球（stripe）
                    
                    Ball 对象属性：
                        ball.state.rvw: np.ndarray, shape=(3,3)
                            [0]: position, np.array([x, y, z])  # 位置，单位：米
                            [1]: velocity, np.array([vx, vy, vz])  # 速度，单位：米/秒
                            [2]: spin, np.array([wx, wy, wz])  # 角速度，单位：弧度/秒
                        
                        ball.state.s: int  # 状态码
                            0 = 静止状态
                            4 = 已进袋（通过 ball.state.s == 4 判断）
                            1-3 = 运动中间状态（滑动/滚动/旋转）
                        
                        ball.state.t: float  # 时间戳，单位：秒
                    
                    示例：
                        pos = balls['cue'].state.rvw[0]  # 白球位置
                        pocketed = (balls['1'].state.s == 4)  # 1号球是否进袋
                
                my_targets (list[str]): 该玩家的目标球ID列表
                    - 正常情况：['1', '2', ...] 或 ['9', '10', ...]
                    - 目标球全部进袋后：['8']（需打黑8）
                
                table (Table): 球桌对象
                    属性：
                        table.w: float  # 球桌宽度，单位：米（约0.99米）
                        table.l: float  # 球桌长度，单位：米（约1.98米）
                        
                        table.pockets: dict, {pocket_id: Pocket对象}
                            pocket_id 取值：
                                'lb', 'lc', 'lt'  # 左侧：下、中、上
                                'rb', 'rc', 'rt'  # 右侧：下、中、上
                            
                            Pocket.center: np.array([x, y, z])  # 球袋中心坐标
                        
                        table.cushion_segments: CushionSegments  # 库边信息
                    
                    示例：
                        width = table.w
                        lb_pos = table.pockets['lb'].center
                        pocket_ids = list(table.pockets.keys())
        """

class _BallParameters:
    """Parameters used in paper"""

    mass: float = 0.1406  # kg
    radius: float = 0.02625  # m
    h: float = 0.03675
    restitution: float = 0.98  # coefficient of restitution
    table_friction: float = 0.212  # sliding friction coefficient between ball and table
    cushion_friction: float = (
        0.14  # sliding friction coefficient between ball and cushion
    )