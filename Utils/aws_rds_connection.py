from sqlalchemy import create_engine

def create_connection(user='admin', pw='Wooji1234', db='wj_Option', endpoint='wjweb.c5ax87iajl4c.us-east-2.rds.amazonaws.com'): 
    sqlEngine = create_engine('mysql+pymysql://{user}:{pw}@{endpoint}:3306/{db}'.format(user=user, pw=pw, endpoint=endpoint, db=db))
    dbConnection = sqlEngine.connect()

    return dbConnection