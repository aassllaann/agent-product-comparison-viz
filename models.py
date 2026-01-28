from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from config import DB_URI

engine = create_engine(DB_URI)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Camera(Base):
    __tablename__ = "cameras"
    id = Column(Integer, primary_key=True, index=True)
    Brand = Column(String)
    Model = Column(String)
    Alias = Column(String)  # New column for search matching
    Price = Column(Float)
    Year = Column(Integer)
    image_file = Column(String)
    Total_megapixels = Column(Float)
    Sensor_type = Column(String)
    Weight_g = Column(Float)
    Max_ISO = Column(Float)
    Screen_Size_in = Column(Float)
    Supports_4K = Column(Boolean)
    Portability_Score = Column(Float)
    LowLight_Score = Column(Float)
    Video_Score = Column(Float)


class Phone(Base):
    """手机数据模型"""
    __tablename__ = "phones"
    id = Column(Integer, primary_key=True, index=True)
    Brand = Column(String)              # 品牌
    Model = Column(String)              # 型号
    Price = Column(Float)               # 价格
    Year = Column(Integer)              # 上市年份
    image_file = Column(String)         # 图片文件
    Storage_GB = Column(Integer)        # 存储空间
    RAM_GB = Column(Integer)            # 运行内存
    Screen_Size_in = Column(Float)      # 屏幕尺寸
    Battery_mAh = Column(Integer)       # 电池容量
    Camera_MP = Column(Float)           # 主摄像素
    Processor = Column(String)          # 处理器
    OS = Column(String)                 # 操作系统
    # 评分维度
    Performance_Score = Column(Float)   # 性能评分
    Camera_Score = Column(Float)        # 拍照评分
    Battery_Score = Column(Float)       # 续航评分
    Value_Score = Column(Float)         # 性价比评分


class Laptop(Base):
    """笔记本电脑数据模型"""
    __tablename__ = "laptops"
    id = Column(Integer, primary_key=True, index=True)
    Brand = Column(String)              # 品牌
    Model = Column(String)              # 型号
    Price = Column(Float)               # 价格
    Year = Column(Integer)              # 上市年份
    image_file = Column(String)         # 图片文件
    Screen_Size_in = Column(Float)      # 屏幕尺寸
    Weight_kg = Column(Float)           # 重量（千克）
    CPU = Column(String)                # 处理器
    GPU = Column(String)                # 显卡
    RAM_GB = Column(Integer)            # 运行内存
    Storage_GB = Column(Integer)        # 存储空间
    Battery_Hours = Column(Float)       # 续航时间
    Category = Column(String)           # 类型：轻薄本/游戏本/商务本
    # 评分维度
    Performance_Score = Column(Float)   # 性能评分
    Portability_Score = Column(Float)   # 便携性评分
    Display_Score = Column(Float)       # 屏幕评分
    Value_Score = Column(Float)         # 性价比评分


class Headphone(Base):
    """耳机数据模型"""
    __tablename__ = "headphones"
    id = Column(Integer, primary_key=True, index=True)
    Brand = Column(String)              # 品牌
    Model = Column(String)              # 型号
    Price = Column(Float)               # 价格
    Year = Column(Integer)              # 上市年份
    image_file = Column(String)         # 图片文件
    Type = Column(String)               # 类型：头戴式/入耳式/半入耳式
    Wireless = Column(Boolean)          # 是否无线
    ANC = Column(Boolean)               # 主动降噪
    Battery_Hours = Column(Float)       # 续航时间
    Driver_mm = Column(Float)           # 驱动单元尺寸
    Impedance_Ohm = Column(Float)       # 阻抗
    # 评分维度
    Sound_Score = Column(Float)         # 音质评分
    Comfort_Score = Column(Float)       # 舒适度评分
    ANC_Score = Column(Float)           # 降噪评分
    Value_Score = Column(Float)         # 性价比评分


class Tablet(Base):
    """平板电脑数据模型"""
    __tablename__ = "tablets"
    id = Column(Integer, primary_key=True, index=True)
    Brand = Column(String)              # 品牌
    Model = Column(String)              # 型号
    Price = Column(Float)               # 价格
    Year = Column(Integer)              # 上市年份
    image_file = Column(String)         # 图片文件
    Screen_Size_in = Column(Float)      # 屏幕尺寸
    Weight_g = Column(Float)            # 重量（克）
    Processor = Column(String)          # 处理器
    RAM_GB = Column(Integer)            # 运行内存
    Storage_GB = Column(Integer)        # 存储空间
    Battery_mAh = Column(Integer)       # 电池容量
    OS = Column(String)                 # 操作系统
    Stylus_Support = Column(Boolean)    # 是否支持手写笔
    # 评分维度
    Performance_Score = Column(Float)   # 性能评分
    Display_Score = Column(Float)       # 屏幕评分
    Productivity_Score = Column(Float)  # 生产力评分
    Value_Score = Column(Float)         # 性价比评分


class Smartwatch(Base):
    """智能手表数据模型"""
    __tablename__ = "smartwatches"
    id = Column(Integer, primary_key=True, index=True)
    Brand = Column(String)              # 品牌
    Model = Column(String)              # 型号
    Price = Column(Float)               # 价格
    Year = Column(Integer)              # 上市年份
    image_file = Column(String)         # 图片文件
    Screen_Size_in = Column(Float)      # 屏幕尺寸
    Battery_Days = Column(Float)        # 续航天数
    Waterproof_Rating = Column(String)  # 防水等级
    OS = Column(String)                 # 操作系统
    Weight_g = Column(Float)            # 重量（克）
    Health_Features = Column(String)    # 健康功能
    # 评分维度
    Battery_Score = Column(Float)       # 续航评分
    Health_Score = Column(Float)        # 健康功能评分
    Smart_Score = Column(Float)         # 智能体验评分
    Value_Score = Column(Float)         # 性价比评分


class BluetoothSpeaker(Base):
    """蓝牙音箱数据模型"""
    __tablename__ = "bluetooth_speakers"
    id = Column(Integer, primary_key=True, index=True)
    Brand = Column(String)              # 品牌
    Model = Column(String)              # 型号
    Price = Column(Float)               # 价格
    Year = Column(Integer)              # 上市年份
    image_file = Column(String)         # 图片文件
    Power_W = Column(Float)             # 功率（瓦）
    Bluetooth_Version = Column(String)  # 蓝牙版本
    Battery_Hours = Column(Float)       # 续航时间
    Waterproof_Rating = Column(String)  # 防水等级
    Weight_g = Column(Float)            # 重量（克）
    Driver_Config = Column(String)      # 单元配置
    # 评分维度
    Sound_Score = Column(Float)         # 音质评分
    Battery_Score = Column(Float)       # 续航评分
    Portability_Score = Column(Float)   # 便携性评分
    Value_Score = Column(Float)         # 性价比评分


class Monitor(Base):
    """显示器数据模型"""
    __tablename__ = "monitors"
    id = Column(Integer, primary_key=True, index=True)
    Brand = Column(String)              # 品牌
    Model = Column(String)              # 型号
    Price = Column(Float)               # 价格
    Year = Column(Integer)              # 上市年份
    image_file = Column(String)         # 图片文件
    Screen_Size_in = Column(Float)      # 屏幕尺寸
    Resolution = Column(String)         # 分辨率
    Refresh_Rate_Hz = Column(Integer)   # 刷新率
    Panel_Type = Column(String)         # 面板类型
    Color_Gamut = Column(String)        # 色域覆盖
    Response_Time_ms = Column(Float)    # 响应时间
    HDR_Support = Column(String)        # HDR支持
    Ports = Column(String)              # 接口类型
    # 评分维度
    Display_Score = Column(Float)       # 画质评分
    Performance_Score = Column(Float)   # 性能评分
    Ergonomics_Score = Column(Float)    # 人体工学评分
    Value_Score = Column(Float)         # 性价比评分


class GamingConsole(Base):
    """游戏主机数据模型"""
    __tablename__ = "gaming_consoles"
    id = Column(Integer, primary_key=True, index=True)
    Brand = Column(String)              # 品牌
    Model = Column(String)              # 型号
    Price = Column(Float)               # 价格
    Year = Column(Integer)              # 上市年份
    image_file = Column(String)         # 图片文件
    Storage_GB = Column(String)         # 改为 String 以支持 "无" (如串流设备)
    Max_Resolution = Column(String)     # 最高分辨率
    Exclusive_Games_Count = Column(String)  # 改为 String 以支持 "50+", "Steam库" 等内容
    Subscription_Service = Column(String)    # 订阅服务
    Backward_Compatible = Column(String)     # 向下兼容
    # 评分维度
    Performance_Score = Column(Float)   # 游戏性能评分
    Ecosystem_Score = Column(Float)     # 游戏生态评分
    Media_Score = Column(Float)         # 多媒体功能评分
    Value_Score = Column(Float)         # 性价比评分


class GPU(Base):
    """显卡数据模型"""
    __tablename__ = "gpus"
    id = Column(Integer, primary_key=True, index=True)
    Brand = Column(String)              # 品牌
    Model = Column(String)              # 型号
    Price = Column(Float)               # 价格
    Year = Column(Integer)              # 上市年份
    image_file = Column(String)         # 图片文件
    VRAM_GB = Column(Integer)           # 显存容量
    Chip = Column(String)               # 芯片型号
    TDP_W = Column(Integer)             # 功耗
    Recommended_PSU_W = Column(Integer) # 推荐电源
    Memory_Type = Column(String)        # 显存类型
    ThreeDMark_Score = Column(Integer)  # 3DMark跑分 (保留列名为兼容性)
    # 评分维度
    Gaming_Score = Column(Float)        # 游戏性能评分
    Creative_Score = Column(Float)      # 创作性能评分
    Thermal_Score = Column(Float)       # 功耗散热评分
    Value_Score = Column(Float)         # 性价比评分
