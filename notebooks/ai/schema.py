from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.types import Float

Base = declarative_base()

class AgentParams(Base):
    __tablename__ = 'agent_params'

    id = Column(Integer, primary_key=True)

    target = Column(Float)
    learning_rate = Column(Float)
    population_size = Column(Integer)
    sigma = Column(Float)
    size_network = Column(Integer)
    skip = Column(Integer)
    window_size = Column(Integer)
    market = Column(String)
