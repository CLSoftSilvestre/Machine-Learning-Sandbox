
import sqlite3
from sqlite3 import Error
from datetime import datetime
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
    
    # Create the predictions table
    cur.execute('''CREATE TABLE IF NOT EXISTS predictions
                (id integer PRIMARY KEY AUTOINCREMENT, timestamp datetime NOT NULL, model text NOT NULL, status integer NOT NULL, type integer NOT NULL)''')
    
    # Create the operations table
    cur.execute('''CREATE TABLE IF NOT EXISTS operations
                (id integer PRIMARY KEY AUTOINCREMENT, timestamp datetime NOT NULL, user text NOT NULL, type text NOT NULL, description text)''')

    adminPassword = b'admin'
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(adminPassword, salt)

    adminUser = (1, 'admin', hashed, 'Administrator')
    powerUser = (2, 'root', hashed, 'PowerUser')

    sql = '''INSERT INTO users VALUES (?,?,?,?)'''

    cur.execute(sql, adminUser)
    cur.execute(sql, powerUser)

    con.commit()

    # Add operation of creationg database
    data = (1, datetime.now(), "root", "DB CREATION","Creation of main database")
    sql = '''INSERT INTO operations VALUES (?,?,?,?,?)'''
    cur.execute(sql, data)

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
        #create_database(db_file)
    
    return conn

def query(conn, sql, args):
    cur = conn.cursor()
    cur.execute(sql, args)

    rows = cur.fetchall()

    return rows

def UserLogin(db_file, loginData):
    conn = create_connection(db_file)
    user = query(conn, "SELECT * FROM users WHERE name=?", (loginData[0],))
    
    if len(user) > 0:

        if bcrypt.hashpw(loginData[1].encode('ASCII'), user[0][2]) == user[0][2]:
            name = user[0][1]
            role = user[0][3]
            tempUser = User(name, role)
            return tempUser
        else:
            return False
    else:
        return False

def add_Prediction(db_file, timestamp, model, status, type):
    con = sqlite3.connect(db_file)
    cur = con.cursor()

    data = (timestamp, model, status, type)

    sql = '''INSERT INTO predictions(timestamp,model,status,type) VALUES(?,?,?,?)'''
    cur.execute(sql, data)
    con.commit()
    con.close()

def add_Operation(db_file, timestamp, user, type, description):
    con = sqlite3.connect(db_file)
    cur = con.cursor()

    data = (timestamp, user, type, description)

    sql = '''INSERT INTO operations(timestamp,user,type,description) VALUES(?,?,?,?)'''
    cur.execute(sql, data)
    con.commit()
    con.close()


## TEST CODE
#create_database(r'C:\Users\CSilvestre\Code\Machine-Learning-Sandbox\app\database\mls.db')

#userLoginData = ('admin','admin')
#userLoginData = ('pu','pu')
#user = UserLogin(r'C:\Users\CSilvestre\Code\Machine-Learning-Sandbox\app\database\mls.db', userLoginData)






