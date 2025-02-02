import pandas as pd


# Load the data
try:
    data = pd.read_csv('data.csv', dtype=float, sep=',')

except Exception as e:
    exit(e)

# Initialize t0 and t1
t0 = 0
t1 = 0

X = data.iloc[0:len(data), 0]
Y = data.iloc[0:len(data), 1]


try:
    f = open("teta.txt")
    t0 = float(f.readline())
    t1 = float(f.readline())

except Exception:
    print("No training value upload yet")

# Enter mileage to estimate
while True:
    try:
        mileAge_to_estimate = float(input("Enter the mileage of your car:\n"))
        if mileAge_to_estimate < 0:
            print("Please enter a positive value.")
        else:
            break
    except ValueError:
        print("Invalid input. Please enter a numeric value.")

result = t0 + (t1 * mileAge_to_estimate)
print("Estimated Price:")
print(f"{result:.2f}$")
