from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from config import DB_URI

engine = create_engine(DB_URI)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Camera(Base):
    __tablename__ = "cameras"
    id = Column(Integer, primary_key=True, index=True)
    Brand = Column(String); Model = Column(String); Price = Column(Float)
    Weight_g = Column(Float); Max_ISO = Column(Float); Min_Aperture_F = Column(Float)
    Supports_4K = Column(Boolean); Portability_Score = Column(Float)
    LowLight_Score = Column(Float); Video_Score = Column(Float)
    Screen_Size_in = Column(Float)