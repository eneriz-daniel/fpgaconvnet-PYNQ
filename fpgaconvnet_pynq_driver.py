# Copyright (C) 2023 Daniel Enériz
# 
# This file is part of fpgaConvNet PYNQ driver.
# 
# fpgaConvNet PYNQ driver is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# fpgaConvNet PYNQ driver is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with fpgaConvNet PYNQ driver.  If not, see <http://www.gnu.org/licenses/>.

"""
fpgaconvnet_pynq_driver.py
~~~~~~~~~~~~~~~~~~~~~~~~~~

This module contains the fpgaConvNet PYNQ driver.

Author: Daniel Enériz
Version: 1.0
Date: 2023-10-12

Usage:
>>> from pynq import Overlay
>>> from fpgaconvnet_pynq_driver import *
>>> overlay = Overlay('bitstream.bit')
>>> fpgaconvnet_ip = overlay.fpgaconvnet_ip_0
>>> fpgaconvnet_ip.load_partition('partitions.json', 0)
>>> Y = fpgaconvnet_ip.run(X)
"""

# TODOs:
# - Add support for weights reloading
# - Add support for arbitrary number format to be used in the input and output
# - Multiple partitions? This will require to create a higher level class
#   that will manage the partitions bitstreams, jsons, weights, intermediate
#   buffers, etc.
# - At the moment, INT and 1V2 rails are used to measure the PL power. This
#   doesn't have to be the case for other boards. Maybe a dictionary with the
#   rails to be used for each board?

import pynq
import json
from pynq.buffer import allocate
import numpy as np
from time import time, sleep
import os

# Detect if the board has PMBus to measure power
PW_COMPATIBLE_PLATFORMS = ['ZCU104'] # List of boards with PMBus
PW_ENABLED = False
if 'BOARD' in os.environ:
    if os.environ['BOARD'] in PW_COMPATIBLE_PLATFORMS:
        PW_ENABLED = True

class FpgaConvNetDriver(pynq.DefaultIP):

    def __init__(self, description):
        """Driver for the FpgaConvnet IP cores

        Args:
            description (str): Description of the IP core
        """
        super().__init__(description=description)
    
    bindto = ['xilinx.com:hls:fpgaconvnet_ip:1.0']
    
    def load_partition(self, partitions_json, partition_id):
        """Loads the partition with the given ID from the JSON file. Uses the
        information in the partition to allocate the input and output buffers.
        Also sets the mode to 0 (i.e. processing mode)

        Args:
            partitions_json (str): Path to the JSON file containing the partitions description
            partition_id (int): ID of the partition to load
        
        Raises:
            FileNotFoundError: If the JSON file is not found
            ValueError: If the partition ID is not found in the JSON file
        """

        # Load the JSON file
        with open(partitions_json, 'r') as f:
            partitions = json.load(f)

        # Find the partition with the given ID
        partition = None
        for p in partitions['partition']:
            if int(p['id']) == partition_id:
                partition = p
                break
        
        # Raise an error if the partition ID is not found
        if partition is None:
            raise ValueError(f"Partition ID {partition_id} not found in {partitions_json}")

        self.partition_id = partition_id
        self.partition = partition

        # Parse the partition dictionary to get needed information
        self._parse_partition(partition)

        # Allocate buffers for the input and output
        self._allocate_buffers()

        # Set the input and output buffers
        self.register_map.fpgaconvnet_in_0 = self.input_buffer.physical_address
        self.register_map.fpgaconvnet_out_0 = self.output_buffer.physical_address

        # Set mode to 0 (process data)
        self.register_map.mode = 0

    def _parse_partition(self, partition):
        """Reads the partition needed for the driver.
        
        Args:
            partition (dict): Dictionary containing the partition details

        Raises:
            AssertionError: If the coarse in or out factors are not between 1 and 4
        """

        # Get the batch size
        self.batch_size = partition['layers'][0]['parameters']['batch_size']

        # Get input shape (rows, cols, channels) and the coarse in factor
        self.rows_in = partition['layers'][0]['parameters']['rows_in']
        self.cols_in = partition['layers'][0]['parameters']['cols_in']
        self.channels_in = partition['layers'][0]['parameters']['channels_in']
        self.coarse_in = partition['layers'][0]['parameters']['coarse_in']

        # Get output shape (rows, cols, channels) and the coarse out factor
        self.rows_out = partition['layers'][-1]['parameters']['rows_out']
        self.cols_out = partition['layers'][-1]['parameters']['cols_out']
        self.channels_out = partition['layers'][-1]['parameters']['channels_out']
        self.coarse_out = partition['layers'][-1]['parameters']['coarse_out']

        # Ensure that the coarse in and out factors are between 1 and 4
        assert self.coarse_in >= 1 and self.coarse_in <= 4, \
            "Coarse in factor must be between 1 and 4"
        assert self.coarse_out >= 1 and self.coarse_out <= 4, \
            "Coarse out factor must be between 1 and 4"
        
    def _allocate_buffers(self):
        """Allocates buffers for the input and output.

        The format of input and output buffers in fpgaconvnet is based in 64-bit
        words. Each 64-bit word can contain up to 4 16-bit data elements. The
        coarse factor determines how many of these 16-bit data elements are used
        in the buffer. For example, if the coarse factor is 1, then the 64-bit
        word will contain 1 16-bit data element with the format
        `0x0000 0000 0000 XXXX`; if the coarse factor is 2, then the 64-bit word
        will contain 2 16-bit data elements with the format
        `0x0000 0000 XXXX XXXX`; and so on. Thus the number of 64-bit words that
        must be allocated is the number of input/output elements divided by the
        respective coarse factor.
        """

        # Allocate input buffer
        self.input_buffer = allocate(
            shape=(self.batch_size*self.rows_in*self.cols_in*self.channels_in//self.coarse_in,),
            dtype='u8' # 64-bit unsigned integer
        )

        # Allocate output buffer
        self.output_buffer = allocate(
            shape=(self.batch_size*self.rows_out*self.cols_out*self.channels_out//self.coarse_out,),
            dtype='u8' # 64-bit unsigned integer
        )
        
    def _write_input(self, X):
        """Writes the input to the input buffer.

        The format of input and output buffers in fpgaconvnet is based in 64-bit
        words. Each 64-bit word can contain up to 4 16-bit data elements. The
        coarse factor determines how many of these 16-bit data elements are used
        in the buffer. For example, if the coarse factor is 1, then the 64-bit
        word will contain 1 16-bit data element with the format
        `0x0000 0000 0000 XXXX`; if the coarse factor is 2, then the 64-bit word
        will contain 2 16-bit data elements with the format
        `0x0000 0000 XXXX XXXX`; and so on. Thus the number of 64-bit words that
        must be allocated is the number of input/output elements divided by the
        respective coarse factor.

        Args:
            X (numpy.ndarray): Input data
        """

        # Ensure that the input shape matches the expected shape
        assert X.shape == (self.batch_size, self.channels_in, self.rows_in,
                           self.cols_in), \
            f"Input shape {X.shape} does not match expected shape " \
            f"{(self.batch_size, self.channels_in, self.rows_in, self.cols_in)}. " \
            "Remember that the input format is (batch_size, channels, rows, cols)."

        arr = np.multiply(X, 256).astype('i2') # 16-bit signed integer
        arr = arr.astype('u2') # 16-bit unsigned integer

        # Flatten the input
        arr = arr.flatten('C')

        # Fill the input with zeros to ensure there are coarse_in  elements with
        # data per each 4 elements
        expanded_arr = np.zeros(self.input_buffer.size*4, dtype='u2') # 16-bit unsigned integer
        for i in range(self.coarse_in):
            expanded_arr[i::4] = arr[i::self.coarse_in]

        # Transform the input to 64-bit unsigned integer. This will group the 4
        # 16-bit elements into a single 64-bit element
        arr = np.frombuffer(expanded_arr.tobytes(), dtype='u8') # 64-bit unsigned integer

        # Write the input to the input buffer
        self.input_buffer[:] = arr

        # Flush the input buffer to ensure that the data is written to the
        # physical memory
        self.input_buffer.flush()

    def _read_output(self):
        """Reads the output from the output buffer.

        The format of input and output buffers in fpgaconvnet is based in 64-bit
        words. Each 64-bit word can contain up to 4 16-bit data elements. The
        coarse factor determines how many of these 16-bit data elements are used
        in the buffer. For example, if the coarse factor is 1, then the 64-bit
        word will contain 1 16-bit data element with the format
        `0x0000 0000 0000 XXXX`; if the coarse factor is 2, then the 64-bit word
        will contain 2 16-bit data elements with the format
        `0x0000 0000 XXXX XXXX`; and so on. Thus the number of 64-bit words that
        must be allocated is the number of input/output elements divided by the
        respective coarse factor.

        Returns:
            numpy.ndarray: Output data
        """

        Y = self.output_buffer

        Y = np.frombuffer(Y.tobytes(), dtype='u2') # 16-bit unsigned integer

        # Remove the zeros added due to the coarse factor
        arr = np.zeros(self.output_buffer.size*self.coarse_out, dtype='u2') # 16-bit unsigned integer
        for i in range(self.coarse_out):
            arr[i::self.coarse_out] = Y[i::4]
        
        # Reshape the output to the expected shape
        Y = arr.reshape(self.batch_size, self.channels_out, self.rows_out, self.cols_out)

        # Transform the output to float
        Y = Y.astype('i2').astype('float32')
        Y = np.divide(Y, 256)

        return Y
    
    def run(self, X):
        """Runs the partition with the given input.

        Args:
            X (numpy.ndarray): Input data
        
        Returns:
            numpy.ndarray: Output data
        """

        # Write the input to the input buffer
        self._write_input(X)

        # Send the AP_START signal
        self.register_map.CTRL.AP_START = 1

        # Wait until the AP_DONE signal is received
        while self.register_map.CTRL.AP_DONE == 0:
            pass

        # Read the output from the output buffer
        return self._read_output()

    def get_latency(self, X, num_runs=1024):
        """Get the latency of the IP by running it multiple times.

        Args:
            X (np.array): Input data to be used for the test.
            num_runs (int, optional): Number of runs to be performed. Defaults
                to 1024.
        
        Returns:
            float: Latency in seconds
        """

        # Write the input to the input buffer
        self._write_input(X)

        # Run the IP multiple times
        # Get the initial time
        t0 = time()
        for i in range(num_runs):
            # Send the AP_START signal
            self.register_map.CTRL.AP_START = 1

            # Wait until the AP_DONE signal is received
            while self.register_map.CTRL.AP_DONE == 0:
                pass
    
        # Get the time interval
        t1 = time()
        t = t1 - t0

        self.latency = t/num_runs
        self.throughput = num_runs/t

        return t/num_runs
    
    def get_power(self, X, num_runs=1024, off_time=3, track_rails=None, csv_path=None):
        """Get the power, voltage or current values of `track_rails` rails while
        running the IP multiple times. An off time is used to measure prior the
        IP starts running. Note this funtion will only work if the board has
        PMBus.

        Args:
            X (np.array): Input data to be used for the test.
            num_runs (int, optional): Number of runs to be performed. Defaults
                to 1024.
            off_time (int, optional): Time to wait prior to starting the test.
                Allows to measure the power consumption of the board without
                the IP running. Defaults to 3.
            track_rails (list, optional): List of rails to be tracked. Defaults
                to None, in which case all the rails are tracked.
            csv_path (str, optional): Path to the CSV file where the power
                measurements will be saved. Defaults to None, in which case the
                measurements will not be saved.
        
        Returns:
            pd.DataFrame: Dataframe containing the power measurements of all the
                rails available in the board.
        """

        if not PW_ENABLED:
            return None

        if track_rails is None:
            track_rails = []
            rails = pynq.get_rails()
            for rail in rails:
                track_rails.append(rails[rail].voltage)
                track_rails.append(rails[rail].current)
                track_rails.append(rails[rail].power)
        
        track_rails = [rail for rail in track_rails if rail is not None]

        recorder = pynq.DataRecorder(*track_rails)
        
        with recorder.record(1e-3):

            sleep(off_time)

            recorder.mark()

            for i in range(num_runs):
                # Send the AP_START signal
                self.register_map.CTRL.AP_START = 1

                # Wait until the AP_DONE signal is received
                while self.register_map.CTRL.AP_DONE == 0:
                    pass

        if csv_path is not None:
            recorder.frame.to_csv(csv_path, index_label='Timestamp')
        
        return recorder.frame

    def _print_performance(self):
        """Prints the performance of the IP using the adequate units.
        """
        latency = self.latency
        throughput = self.throughput

        if latency < 1e-6:
            latency = latency*1e9
            latency_unit = 'ns'
        elif latency < 1e-3:
            latency = latency*1e6
            latency_unit = 'us'
        elif latency < 1:
            latency = latency*1e3
            latency_unit = 'ms'
        else:
            latency_unit = 's'

        if PW_ENABLED:

            power = self.power
            power_std = self.power_std
            energy = self.energy
            energy_std = self.energy_std

            if power < 1e-3:
                power = power*1e6
                power_std = power_std*1e6
                power_unit = 'uW'
            elif power < 1:
                power = power*1e3
                power_std = power_std*1e3
                power_unit = 'mW'
            else:
                power_unit = 'W'

            if energy < 1e-3:
                energy = energy*1e6
                energy_std = energy_std*1e6
                energy_unit = 'uJ'
            elif energy < 1:
                energy = energy*1e3
                energy_std = energy_std*1e3
                energy_unit = 'mJ'
            else:
                energy_unit = 'J'

        print(f"Latency: {latency:.2f} {latency_unit}")
        print(f"Throughput: {throughput:.2f} inferences/s")

        if PW_ENABLED:

            print(f"Power: {power:.2f} +/- {power_std:.2f} {power_unit}")
            print(f"Energy per inference: {energy:.2f} +/- {energy_std:.2f} {energy_unit}")

    def test_performance(self, X, num_runs=1024, off_time=3, csv_path=None, verbose=True):
        """Test the IP performance by running it multiple times.

        Args:
            X (np.array): Input data to be used for the test.
            num_runs (int, optional): Number of runs to be performed. Defaults
                to 1024.
            off_time (int, optional): Time to wait prior to starting the test.
                Allows to measure the power consumption of the board without
                the IP running. Defaults to 3.
            csv_path (str, optional): Path to the CSV file where the power
                measurements will be saved. Defaults to None, in which case the
                measurements will not be saved. Note that this will only work if
                the board has PMBus.
            verbose (bool, optional): If True, prints the steps of the test and
                the performance. Defaults to True.
        """

        # Get the latency
        if verbose:
            print("Getting latency...", end='')
        self.get_latency(X, num_runs)
        if verbose:
            print("Done")
        
        if PW_ENABLED:

            # Get the power
            if verbose:
                print("Getting power...", end='')

            PL_rails = ['INT', '1V2']
            all_rails = pynq.get_rails()
            track_rails = [all_rails[rail].power for rail in PL_rails + ['12V']]

            pow_df = self.get_power(X, num_runs, off_time, track_rails, csv_path)

            PL_rails = [label+'_power' for label in PL_rails]

            # Sum the PL_rails 
            pow_df['PL_power'] = pow_df[PL_rails].sum(axis=1)

            # Get the average power when Mark = 0 (passive) and Mark = 1 (active)
            pow_pass = pow_df[pow_df['Mark'] == 0]
            pow_act = pow_df[pow_df['Mark'] == 1]

            # Get the average power of the PL in each case and the difference. Use std to get the error
            pl_pass = pow_pass['PL_power'].mean()
            pl_act = pow_act['PL_power'].mean()

            pl_pass_std = pow_pass['PL_power'].std()
            pl_act_std = pow_act['PL_power'].std()

            pl_diff = pl_act - pl_pass

            pl_diff_std = np.sqrt(pl_pass_std**2 + pl_act_std**2)

            self.power = pl_diff
            self.power_std = pl_diff_std
            self.energy = pl_diff*self.latency
            self.energy_std = pl_diff_std*self.latency

            if verbose:
                print("Done")

        if verbose:
            self._print_performance()        