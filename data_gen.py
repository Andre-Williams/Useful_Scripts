import csv
import random
import time
import psutil

def generate():
    xs = 0
    ys = 0

    with open('data.csv', 'w') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()


    while True:
        with open('data.csv', 'a') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            info = {
                    "Degrees": xs,
                    "DTI": ys
            }

            csv_writer.writerow(info)
            print(xs, ys)

            if xs == 360:
                xs = 0
                ys = psutil.cpu_percent() #random.uniform(0.0010, 0.0090)
            else:
                xs += 360/364
                ys = psutil.cpu_percent() #random.uniform(0.0010, 0.0090)

        time.sleep(0.5)
