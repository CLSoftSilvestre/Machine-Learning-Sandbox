
import sqlite3
from sqlite3 import Error

import bcrypt

class User:

    def __init__(self, name, role):
        self.name = name
        self.role = role

def create_database(db_file):
    con = sqlite3.connect(db_file)
    cur = con.cursor()

    # Create the users table
    cur.execute('''CREATE TABLE IF NOT EXISTS users
                (id integer PRIMARY KEY AUTOINCREMENT, name text NOT NULL, password text NOT NULL, role text NOT NULL)''')

    adminPassword = b'admin'
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(adminPassword, salt)

    adminUser = (1, 'admin', hashed, 'Administrator')
    powerUser = (2, 'pu', 'pu', 'PowerUser')

    sql = '''INSERT INTO users VALUES (?,?,?,?)'''

    cur.execute(sql, adminUser)
    cur.execute(sql, powerUser)

    # Save the changes
    con.commit()

    # Release the connection
    con.close()

def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)
    
    return conn

def query(conn, sql, args):
    cur = conn.cursor()
    cur.execute(sql, args)

    rows = cur.fetchall()

    return rows

def UserLogin(db_file, loginData):
    conn = create_connection(db_file)
    #print (loginData)
    user = query(conn, "SELECT * FROM users WHERE name=?", (loginData[0],))
    
    if len(user) > 0:

        print(loginData[1])
        print(user[0][2])

        #hashed = bcrypt.hashpw(loginData[1], user[0][2])

        if bcrypt.hashpw(loginData[1].encode('ASCII'), user[0][2]) == user[0][2]:
            name = user[0][1]
            role = user[0][3]
            tempUser = User(name, role)
            return tempUser
        else:
            return False
    else:
        return False
    

## TEST CODE
#create_database(r'C:\Users\CSilvestre\Code\Machine-Learning-Sandbox\app\database\mls.db')

#userLoginData = ('admin','admin')
#userLoginData = ('pu','pu')
#user = UserLogin(r'C:\Users\CSilvestre\Code\Machine-Learning-Sandbox\app\database\mls.db', userLoginData)






