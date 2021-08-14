import mysql.connector
from datetime import datetime
import pytz

mydb = mysql.connector.connect(host='localhost', user='root', passwd='')

mycursor = mydb.cursor()

sql = "CREATE DATABASE IF NOT EXISTS chatbot"

mycursor.execute(sql)

mydb = mysql.connector.connect(
    host='localhost', user='root', passwd='', database='chatbot')

mycursor = mydb.cursor()
sql = "CREATE TABLE IF NOT EXISTS user(CID INTEGER(8), Password CHAR(30), Name CHAR(60), Mobile_Number CHAR(10), Email CHAR(90), Bank CHAR(9), Account CHAR(15), Amount BIGINT)"
mycursor.execute(sql)

mycursor = mydb.cursor()
sql = "CREATE TABLE IF NOT EXISTS vendor(CID INTEGER(8), PIN INTEGER(4), Vendor CHAR(60), Payment INTEGER)"
mycursor.execute(sql)

mycursor = mydb.cursor()
sql = "CREATE TABLE IF NOT EXISTS subscrip(CID INTEGER(8), PIN INTEGER(4), Monthly_Subs CHAR(60), Payment INTEGER)"
mycursor.execute(sql)

mycursor = mydb.cursor()
sql = "CREATE TABLE IF NOT EXISTS transaction(CID INTEGER(8), Payment_Type INTEGER(1), Payment_Date DATE, Payment_Time TIME, Payment INTEGER)"
mycursor.execute(sql)


# check CID
def findCID(CID):
    mycursor = mydb.cursor()
    sql = """SELECT CID FROM user"""
    mycursor.execute(sql)
    myresult = mycursor.fetchall()
    for results in myresult:
        if CID == results[0]:
            return True
    return False


# check account no.
def findAccount(accountNo):
    mycursor = mydb.cursor()
    sql = """SELECT Account FROM user"""
    mycursor.execute(sql)
    myresult = mycursor.fetchall()
    for results in myresult:
        if accountNo == results[0]:
            return True
    return False


# check Account No. along with CID
def findAcc(CID, accountNo):
    mycursor = mydb.cursor()
    sql = """SELECT CID,Account FROM user"""
    mycursor.execute(sql)
    myresult = mycursor.fetchall()
    for results in myresult:
        if (CID, accountNo) == results:
            return True
    return False


# check Password
def verifyPass(CID, accountNo, password):
    mycursor = mydb.cursor()
    sql = """SELECT CID,Account,Password FROM user"""
    mycursor.execute(sql)
    myresult = mycursor.fetchall()
    for results in myresult:
        if (CID, accountNo, password) == results:
            return True
    return False


# Account Balance
def fetch_balance(CID, password, accountNo):
    mycursor = mydb.cursor()
    sql = """SELECT Amount FROM user WHERE CID='%s' AND Password='%s' AND Account='%s' """ % (
        CID, password, accountNo)
    mycursor.execute(sql)
    myresult = mycursor.fetchone()
    balance = myresult[0]
    return balance


# check vendor name
def check_vendor(CID, vendor):
    vendor = vendor.upper()
    mycursor = mydb.cursor()
    sql = """SELECT CID,Vendor FROM vendor"""
    mycursor.execute(sql)
    myresult = mycursor.fetchall()
    for results in myresult:
        if (CID, vendor) == results:
            return True
    return False


# Vendor to Pay
def fetch_vendor(CID, vendor):
    mycursor = mydb.cursor()
    sql = """SELECT * FROM vendor WHERE CID='%s' AND  Vendor='%s'""" % (
        CID, vendor)
    mycursor.execute(sql)
    myresult = mycursor.fetchall()
    return myresult


# account to vendor transaction
def checkout_vendor(CID, PIN, Vendor, Payment, password, accountNo) -> None:
    mycursor = mydb.cursor()
    sql = """SELECT Payment FROM vendor WHERE CID='%s' AND PIN='%s' AND Vendor='%s'AND Payment='%s'""" % (
        CID, PIN, Vendor, Payment)
    mycursor.execute(sql)
    myresult = mycursor.fetchone()

    mycursor = mydb.cursor()
    sql = """SELECT Amount FROM user WHERE CID='%s' AND Password='%s' AND Account='%s'""" % (
        CID, password, accountNo)
    mycursor.execute(sql)
    myresult = mycursor.fetchone()
    amountI = myresult[0]

    amountF = amountI-Payment

    mycursor = mydb.cursor()
    sql = "UPDATE user SET Amount='%s' WHERE CID='%s' AND Password='%s' AND Account='%s'""" % (
        amountF, CID, password, accountNo)
    mycursor.execute(sql)
    mydb.commit()

    mycursor = mydb.cursor()
    sql = """DELETE FROM vendor WHERE CID='%s' AND PIN='%s' AND Vendor='%s'AND Payment='%s'""" % (
        CID, PIN, Vendor, Payment)
    mycursor.execute(sql)
    mydb.commit()

    today = datetime.now(pytz.timezone('Asia/Kolkata'))
    d1 = today.strftime('%Y-%m-%d')
    d2 = today.strftime('%H:%M:%S')

    mycursor = mydb.cursor()
    sql = "INSERT INTO transaction (CID, Payment_Type, Payment_Date, Payment_Time, Payment) VALUES (%s, %s, %s, %s, %s)"
    val = (CID, 0, d1, d2, Payment)
    mycursor.execute(sql, val)
    mydb.commit()


# Monthly Sub to Pay
def check_monthlySubscription(CID):
    mycursor = mydb.cursor()
    sql = """SELECT * FROM subscrip WHERE CID='%s'""" % (
        CID)
    mycursor.execute(sql)
    myresult = mycursor.fetchall()
    return myresult


# account to monthly payment transaction
def checkout_monthlySubscription(CID, PIN, Monthly_Subs, Payment, password, accountNo) -> None:
    mycursor = mydb.cursor()
    sql = """SELECT Payment FROM subscrip WHERE CID='%s' AND PIN='%s' AND Monthly_Subs='%s'AND Payment='%s'""" % (
        CID, PIN, Monthly_Subs, Payment)
    mycursor.execute(sql)
    myresult = mycursor.fetchone()

    mycursor = mydb.cursor()
    sql = """SELECT Amount FROM user WHERE CID='%s' AND Password='%s' AND Account='%s'""" % (
        CID, password, accountNo)
    mycursor.execute(sql)
    myresult = mycursor.fetchone()
    amountI = myresult[0]

    amountF = amountI-Payment

    mycursor = mydb.cursor()
    sql = "UPDATE user SET Amount='%s' WHERE CID='%s' AND Password='%s' AND Account='%s'""" % (
        amountF, CID, password, accountNo)
    mycursor.execute(sql)
    mydb.commit()

    mycursor = mydb.cursor()
    sql = """DELETE FROM subscrip WHERE CID='%s' AND PIN='%s' AND Monthly_Subs='%s'AND Payment='%s'""" % (
        CID, PIN, Monthly_Subs, Payment)
    mycursor.execute(sql)
    mydb.commit()

    today = datetime.now(pytz.timezone('Asia/Kolkata'))
    d1 = today.strftime('%Y-%m-%d')
    d2 = today.strftime('%H:%M:%S')

    mycursor = mydb.cursor()
    sql = "INSERT INTO transaction (CID, Payment_Type, Payment_Date, Payment_Time, Payment) VALUES (%s, %s, %s, %s, %s)"
    val = (CID, 0, d1, d2, Payment)
    mycursor.execute(sql, val)
    mydb.commit()


# fetch transactions
def show_transactions(CID, password, accountNo):
    mycursor = mydb.cursor()
    sql = """SELECT * FROM transaction WHERE CID='%s' ORDER BY Payment_Date DESC, Payment_Time DESC""" % (
        CID)
    mycursor.execute(sql)
    myresult = mycursor.fetchall()
    return myresult


# account to account transaction
def acc_to_acc(CIDF, password, fromAccountNo, toAccountNo, amount) -> None:
    mycursor = mydb.cursor()
    sql = """SELECT Amount FROM user WHERE CID='%s' AND Password='%s' AND Account='%s'""" % (
        CIDF, password, fromAccountNo)
    mycursor.execute(sql)
    myresult = mycursor.fetchone()
    amountF = myresult[0]

    mycursor = mydb.cursor()
    sql = """SELECT Amount,CID FROM user WHERE Account='%s'""" % (toAccountNo)
    mycursor.execute(sql)
    myresult = mycursor.fetchone()
    amountT = myresult[0]
    CIDTo = myresult[1]

    amountF -= int(amount)
    amountT += int(amount)

    mycursor = mydb.cursor()
    sql = "UPDATE user SET Amount='%s' WHERE CID='%s' AND Password='%s' AND Account='%s'""" % (
        amountF, CIDF, password, fromAccountNo)
    mycursor.execute(sql)
    mydb.commit()

    mycursor = mydb.cursor()
    sql = "UPDATE user SET Amount='%s' WHERE Account='%s'""" % (
        amountT, toAccountNo)
    mycursor.execute(sql)
    mydb.commit()

    today = datetime.now(pytz.timezone('Asia/Kolkata'))
    d1 = today.strftime("%Y-%m-%d")
    d2 = today.strftime("%H:%M:%S")

    mycursor = mydb.cursor()
    sql = "INSERT INTO transaction (CID, Payment_Type, Payment_Date, Payment_Time, Payment) VALUES (%s, %s, %s, %s, %s)"
    val = (CIDF, 0, d1, d2, amount)
    mycursor.execute(sql, val)
    mydb.commit()

    mycursor = mydb.cursor()
    sql = "INSERT INTO transaction (CID, Payment_Type, Payment_Date, Payment_Time, Payment) VALUES (%s, %s, %s, %s, %s)"
    val = (CIDTo, 1, d1, d2, amount)
    mycursor.execute(sql, val)
    mydb.commit()