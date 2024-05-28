import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Define the number of rows for test data
num_rows = 1000

# Generate random dates of birth between 1950-01-01 and 2000-12-31
dob = [datetime.strftime(datetime(1950, 1, 1) + timedelta(days=random.randint(0, 365*50)), '%Y-%m-%d') for _ in range(num_rows)]

# Generate random ages based on dates of birth
today = datetime.today()
age = [today.year - int(dob_str[:4]) - ((today.month, today.day) < (int(dob_str[5:7]), int(dob_str[8:]))) for dob_str in dob]

# Generate random cities and states
cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
states = ['NY', 'CA', 'IL', 'TX', 'AZ']
city = [random.choice(cities) for _ in range(num_rows)]
state = [random.choice(states) for _ in range(num_rows)]

# Generate random number of times they bought
times_bought = [random.randint(1, 10) for _ in range(num_rows)]

# Generate random time intervals between purchases
time_intervals = [0 if tb == 1 else random.uniform(0, 365) for tb in times_bought]  # Set interval to 0 for first buy

# Generate random types of scan received
scan_types = ['Scan A', 'Scan B', 'Scan C', 'Scan D', 'Scan E']
type_of_scan = [random.choice(scan_types) for _ in range(num_rows)]

# Generate random UTM sources
utm_sources = ['Social Media', 'Search Engine', 'Referral', 'Direct', 'Email']
utm = [random.choice(utm_sources) for _ in range(num_rows)]

# Generate random facility locations
facilities = ['Facility {}'.format(i) for i in range(1, 26)]
facility = [random.choice(facilities) for _ in range(num_rows)]

# Generate the repurchase column
repurchase = [1 if tb > 5 else 0 for tb in times_bought]

# Create the DataFrame
test_data = pd.DataFrame({
    'DOB': dob,
    'Age': age,
    'City': city,
    'State': state,
    'Times_Bought': times_bought,
    'Time_Intervals': time_intervals,
    'Type_of_Scan': type_of_scan,
    'UTM': utm,
    'Facility': facility,
    'Repurchase': repurchase  # Add the repurchase column
})

# Save the test data to a CSV file
test_data.to_csv('test_data.csv', index=False)
