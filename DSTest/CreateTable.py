import pymysql

con = pymysql.connect(host='localhost', user='root', password='********', db='db_score')

cur = con.cursor()

sql =  '''
            create table score
                (
                     sno int primary key,
                     attendance float , 
                     homework float ,
                     discussion int ,
                     midterm float ,
                     final float , 
                     score float,
                     grade varchar(2) );
          '''

cur.execute(sql)

con.commit()
con.close()

