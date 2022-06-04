import msql as msql
import pandas as pd
import mysql.connector as mysql  # connecting with mysql
from mysql.connector import Error


def database(doc):
    try:
        conn = mysql.connect(host='localhost', database='heart_risk_prediction', user='root', password='')
        if conn.is_connected():
            cursor = conn.cursor()
            cursor.execute("select database();")
            record = cursor.fetchone()
            print("You're connected to database: ", record)

            # creating  table :attendance_data
            cursor.execute('DROP TABLE IF EXISTS cardiologist_details;')
            print('Creating table...')

            # passing the create table statement which  want (column names)
            cursor.execute(
                "CREATE TABLE cardiologist_details(doctorID varchar(255),doctorName varchar(255),email varchar(255))")
            print("Table is created....")

            # # looping  through the data frame(attendance data)
            for x, row in doc.iterrows():
                sql = "INSERT INTO heart_risk_prediction.cardiologist_details VALUES (%s,%s,%s)"
                cursor.execute(sql, tuple(row))
                print("Record inserted to the database table")
                conn.commit()
            sql = "SELECT * FROM heart_risk_prediction.cardiologist_details"
            cursor.execute(sql)
            # # Fetching  all the records in the student_Attendance_25-08-2021.csv
            result = cursor.fetchall()
            for x in result:
                print(x)

            # returning message
            return 'Cardiologist Table successfully created'
    except Error as e:
        print("Error while connecting to MySQL", e)


def read_csv(path, filename):
    cardio = pd.read_csv(path + filename, index_col=False, delimiter=',')
    print(cardio.head())
    return cardio


# calling main method and command line arguments
if __name__ == '__main__':
    file_path = 'D:/python/Cardio/'
    csv_filename = 'cardiologist_details.csv'
    doctors = read_csv(file_path, csv_filename)
    status = database(doctors)
    print(status)

