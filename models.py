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