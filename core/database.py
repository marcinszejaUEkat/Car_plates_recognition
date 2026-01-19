from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

SQLALCHEMY_DATABASE_URL = "sqlite:///./plates.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class PlateRecord(Base):
    __tablename__ = "plates"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    detected_text = Column(String)
    confidence = Column(Float)
    processed_at = Column(DateTime, default=datetime.now)
    status = Column(String)

# Tworzenie tabel
Base.metadata.create_all(bind=engine)

def save_result(filename, text, conf, status="PROCESSED"):
    session = SessionLocal()
    record = PlateRecord(
        filename=filename,
        detected_text=text,
        confidence=conf,
        status=status
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    session.close()
    return record