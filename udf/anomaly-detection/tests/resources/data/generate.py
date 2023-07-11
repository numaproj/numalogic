import json
import time
import random


def create_data(start_time, end_time, filename):
    data = []
    current_time = start_time
    while current_time < end_time:
        timestamp = str(int(current_time * 1000))
        # unique_key = "service-mesh-s2s:6055107291188110321:" + str(random.randint(0, 10**19))
        unique_key = "service-mesh-s2s:6055107291188110321"
        error_rate = str(round(random.uniform(0, 100), 2))
        data.append({"timestamp": timestamp, "unique_key": unique_key, "error_rate": error_rate})
        current_time += 60  # increase timestamp value by 1 min

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    return data


# example usage:
start_time = time.time()  # current UTC timestamp in seconds
end_time = start_time + 360  # 1 hour later
data = create_data(start_time, end_time, "stream2.json")
print(data)  # output the generated JSON output the generated JSON
