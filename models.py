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