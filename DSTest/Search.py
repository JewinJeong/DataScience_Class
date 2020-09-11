import pymysql

conn = pymysql.connect(host = 'localhost', user = 'root', password= 'jj8575412', db = 'db_score')
curs = conn.cursor()

sql = """select sno, midterm, final from score where midterm >= 20 and final >= 20
        order by sno
        """

curs.execute(sql)
rows = curs.fetchone()

while rows:
    print(rows)
    rows = curs.fetchone()

curs.close()
conn.close()


