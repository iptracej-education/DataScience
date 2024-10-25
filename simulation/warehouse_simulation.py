# Warehouse simulation project 
# Written by: Kiyoshi Watanabe  
# 2024-10-24 20:44:19
# 
# The purpose of this simulation is to model a warehouse fulfillment process, 
# where orders are received, picked from inventory, and packed for shipment. 
# The goal is to analyze the system’s performance based on varying numbers of 
# pickers and packers, and to determine the optimal configuration that minimizes 
# the total average wait time for orders while using the least amount of resources 
# (pickers and packers).

# Orders arrive according to a Poisson distribution, which models random,
# independent arrivals of customer orders. In this simulation:
# Order Arrival Rate (λ): Orders arrive every 0.02 minutes on average, which corresponds 
# to 50 orders per minute (i.e., a busy warehouse).
# This simulates the random nature of customer orders being placed at the warehouse.

# Order Processing Stages
# Each order goes through two key stages: Picking and Packing
# After an order arrives, it is first assigned to a picker. Pickers are the workers or 
# robots who retrieve items from the warehouse inventory based on the order.
# Processing Time: The time it takes to pick an order is modeled as an exponential 
# distribution with a mean time of 0.75 minutes. This distribution models the variability 
# in how long it takes to pick items for each order.

# Once the items are picked, the order is assigned to a packer, who prepares the items 
# for shipment
# Processing Time: The packing time is modeled using a uniform distribution between 
# 0.5 and 1 minute, simulating the even spread of packing times depending on the 
# complexity of the order and the number of items to pack.

# The simulation tracks the waiting times in both the picking and packing queues, and 
# computes the average waiting time for each stage, as well as the total average wait time.

# After running the simulation for various combinations of pickers and packers, the system 
# outputs the optimal number of pickers and packers that meet the target average wait time 
# of 60 minutes, while minimizing the total number of resources (pickers + packers).

# The goal is to test different configurations (e.g., 1 to 51 pickers and 11 to 51 packers) 
# to ensure that the total average wait time is under 60 minutes, and to find the configuration 
# that uses the fewest resources while achieving this goal.

# How to run
# >python warehouse_simulation.py
# (wait for a few minutes)

import simpy
import random
import numpy as np
import pandas as pd

order_interval = 0.2  # Orders arrive every 0.2 minutes
# order_interval = 0.02  # High volume of orders
    
sim_duration = 960  # Run the simulation for 960 minutes (16 hours)
TARGET_AVERAGE_WAIT_TIME = 60  # Target average wait time (15 minutes)

# Order class (formerly Passenger)
class Order:
    def __init__(self, order_id, order_type, order_data):
        self.order_id = order_id
        self.order_type = order_type
        self.order_data = order_data  # Combined data with probability and processing times
    
    def get_picking_time(self):
        # Return the exponential time for picking based on the order type
        mean_time = self.order_data[self.order_type]['picking_time']
        return random.expovariate(1.0 / mean_time)  # Exponential distribution

    def get_packing_time(self):
        # Return the uniformly distributed time for packing
        time_range = self.order_data[self.order_type]['packing_times']
        return random.uniform(time_range[0], time_range[1])

# Warehouse Simulation class (formerly AirportSimulation)
class WarehouseSimulation:
    def __init__(self, env, num_pickers, num_packers, order_interval, sim_duration, order_data):
        self.env = env
        self.num_pickers = num_pickers
        self.num_packers = num_packers
        self.order_interval = order_interval
        self.sim_duration = sim_duration
        self.order_data = order_data  # Combined data with probabilities and times

        # Define resources
        self.pickers = simpy.Resource(self.env, num_pickers)
        self.packers = [simpy.Resource(self.env, capacity=1) for _ in range(num_packers)]

        # Tracking total wait times for picking and packing
        self.picking_wait_times = []
        self.packing_wait_times = []

    def run_simulation(self):
        self.env.process(self.generate_orders())  # Start generating orders
        self.env.run(until=self.sim_duration)  # Start the simulation and run until the end
        return self.summarize_results()  # Return the results after simulation ends

    def generate_orders(self):
        order_id = 0
        while True:
            order_type = self.choose_order_type()  # Choose based on probability
            order = Order(order_id, order_type, self.order_data)  # Create order object
            self.env.process(self.order_process(order))  # Start the order's process
            order_id += 1
            yield self.env.timeout(random.expovariate(1.0 / self.order_interval))  # Poisson distributed arrivals

    def choose_order_type(self):
        types = list(self.order_data.keys())
        probabilities = [self.order_data[otype]['probability'] for otype in types]
        return random.choices(types, weights=probabilities, k=1)[0]

    def order_process(self, order):
        # Picking Process
        picking_arrival_time = self.env.now
        with self.pickers.request() as request:
            yield request
            picking_wait_time = self.env.now - picking_arrival_time  # Calculate waiting time
            self.picking_wait_times.append(picking_wait_time)

            # Get the picking time from the order object (exponential distribution)
            yield self.env.timeout(order.get_picking_time())  
        
        # Packing Process (Shortest queue)
        packing_arrival_time = self.env.now
        shortest_packer = min(self.packers, key=lambda x: len(x.queue))
        with shortest_packer.request() as request:
            yield request
            packing_wait_time = self.env.now - packing_arrival_time  # Calculate waiting time
            self.packing_wait_times.append(packing_wait_time)

            # Get the packing time from the order object (uniform distribution)
            yield self.env.timeout(order.get_packing_time())  

    def summarize_results(self):
        avg_picking_wait = np.mean(self.picking_wait_times) if self.picking_wait_times else 0
        avg_packing_wait = np.mean(self.packing_wait_times) if self.packing_wait_times else 0
        total_avg_wait_time = avg_picking_wait + avg_packing_wait
        
        return {
            "avg_picking_wait": avg_picking_wait,
            "avg_packing_wait": avg_packing_wait,
            "total_avg_wait_time": total_avg_wait_time
        }

# Main function
if __name__ == "__main__":

    results = []

    env = simpy.Environment()


    # Updated order data with separate process times for picking and packing
    order_data = {
        'regular': {
            'probability': 1.0,             # 100% of orders will be 'regular'
            'picking_time': 0.75,           # Mean time for picking (exponential distribution)
            'packing_times': (0.5, 1.0)     # Uniform time for packing between 0.5 and 1.0 minutes
        }
    }

    # Test combinations for pickers and packers in the range of 35 to 42 for pickers and 31 to 51 for packers
    for num_pickers in range(1, 50, 1):  # 1 to 50 pickers
        for num_packers in range(1, 10, 1):  # 1 to 50 packers
            env = simpy.Environment()
            warehouse_sim = WarehouseSimulation(env, num_pickers, num_packers, order_interval, sim_duration, order_data)
            result = warehouse_sim.run_simulation()

            print(f"Pickers: {num_pickers}, Packers: {num_packers}")
            print(f"Average Picking Wait Time: {result['avg_picking_wait']:.2f} minutes")
            print(f"Average Packing Wait Time: {result['avg_packing_wait']:.2f} minutes")
            print(f"Total Average Wait Time: {result['total_avg_wait_time']:.2f} minutes\n")
        
            results.append({
                'Pickers': num_pickers,
                'Packers': num_packers,
                'Average_Picking_Wait': result['avg_picking_wait'],
                'Average_Packing_Wait': result['avg_packing_wait'],
                'Total_Average_Wait_Time': result['total_avg_wait_time']
            })
             
    # Filter the results to only include those that meet the target wait time
    valid_results = [result for result in results if result['Total_Average_Wait_Time'] < TARGET_AVERAGE_WAIT_TIME]

    # If there are valid results, find the one with the minimum total resources
    if valid_results:
        # Add a new field 'Total_Resources' as the sum of Pickers and Packers
        for result in valid_results:
            result['Total_Resources'] = result['Pickers'] + result['Packers']
        
        # Step 1: Find the minimum value for 'Total_Resources'
        min_resources = min(valid_results, key=lambda x: x['Total_Resources'])['Total_Resources']

        # Step 2: Filter all results that have the same 'Total_Resources'
        optimal_results = [result for result in valid_results if result['Total_Resources'] == min_resources]

        # Print all optimal results
        print(f"\nFound {len(optimal_results)} optimal configurations with {min_resources} total resources:")
        for result in optimal_results:
            print(f"Pickers: {result['Pickers']}, Packers: {result['Packers']}, "
                  f"Total Resources: {result['Total_Resources']}, Total Average Wait Time: {result['Total_Average_Wait_Time']:.2f} minutes")
    else:
        print("No setup meets the target wait time.")

    # Write results to a CSV file
    df = pd.DataFrame(results)
    csv_file = 'simulation_results.csv'

    try:
        # Attempt to write to the CSV file
        df.to_csv(csv_file, index=False)
        print(f"Results successfully saved to '{csv_file}'.")

    except PermissionError:
        # Catch the specific error if the file is open in another application
        print(f"Error: The file '{csv_file}' is open in another application (e.g., Excel). Please close it and try again.")

    except Exception as e:
        # Catch any other exceptions that might occur
        print(f"An unexpected error occurred: {e}")
