import spam
import os
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
engine = create_engine(os.environ[NameProjekt])
Session = sessionmaker(bind=engine)
#init file
