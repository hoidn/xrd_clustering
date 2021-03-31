'''
Courtesy of Alexander Ivananov, alexander_ivanov@brown.edu

DATA GRID

Object used to represent measurements on a grid and several functions to
convert coordinate systems and find neighbors.

Specifically implemented with 177 point grid as in TiNiSn data set.

Refer to the data_grid_TiNiSn object header for more details on input Regex

'''
import pandas as pd
import numpy as np
import os
import re
import sys

class DataGrid:


    """
    WARNING
    RANGE ONLY WORKS FOR DATA WHERE FIRST COLUMN IS X-AXIS

    """
    def __init__(self, path, regex, range=None, verbose=False):
        """
        Load data and initialize data grid

        # path - where data is loaded from
        # regex - how to parse the grid locations from the file names
        # range - [optional] specific range on the x-axis to consider

        """
        files = os.listdir(path)
        #regex to parse grid location from file
        pattern = re.compile(regex, re.VERBOSE)

        if len(files) == 0:
            print("no files in path")
            sys.exit()

        print("Loading Data from: " + path)
        #load csv files into dictionary
        self.data ={}
        for file in files:
            match = pattern.match(file)
            if(match == None):
                continue
            num = int(match.group("num"))
            if verbose:
                print("\t:: " + str(num) + "\t| " + file)

            data_array = np.array(pd.read_csv(path + file,header=None))
            try:
                data_array[0].astype(np.float)
            except:
                data_array = data_array[1:]
            self.data[num] = data_array.astype(np.float)

        #set of key locations
        self.grid_locations = list(self.data.keys())
        if verbose:
            print("Locations: " + str(self.grid_locations.sort()))

        if len(self.grid_locations) < 177:
            print("# WARNING: missing data files")

        print("Data Loaded Succesfully")

        #number of grid locations
        self.size = len(self.data.keys())
        #length of each spectra vector
        self.data_length = len(self.data[list(self.data.keys())[0]])
        #grid dimentions
        self.dims = (15,15)


        #range is of the form [min,max]
        self.range = range
        self.x_axis = self.data[self.grid_locations[0]][:,0]

        #indices for range start and end
        if self.range == None:
            self.min_index = 0
            self.max_index = self.data_length+1
        else:
            self.min_index = next(i for i,v in enumerate(self.x_axis) if v >= self.range[0])
            self.max_index = next(i-1 for i,v in enumerate(self.x_axis) if v > self.range[1])

        #the right most, left most, bottom most, and top most grid
        #locations in the rows and columns respectively.
        self.row_sums = [5, 14, 25, 38, 51, 66, 81, 96, 111, 126, 139, 152, 163, 172, 177]
        self.row_starts = [1] + [x + 1 for x in self.row_sums[:-1]]
        self.base_vals = [52,26,15,6,7,1,2,3,4,5,13,14,25,38,66]
        self.top_vals = [112,140,153,164,165,173,174,175,176,177,171,172,163,152,126]
        #lengths of each row
        self.row_lengths = [5,9,11,13,13,15,15,15,15,15,13,13,11,9,5]

    """
    Getting the row of a grid locations
    Getting the distance to the location above
    Getting the distance to the location below
    """
    def get_row(self,d):
        return next(i for i,s in enumerate(self.row_sums) if d-s <= 0) + 1
    def up_shift(self,d):
        return (self.row_lengths[self.get_row(d)] + self.row_lengths[self.get_row(d)-1])//2
    def down_shift(self,d):
        return (self.row_lengths[self.get_row(d)-2] + self.row_lengths[self.get_row(d)-1])//2


    def neighbors(self,d):
        """
        Get the neighbors of a grid location
        Returned as dictionary to parse specific neighbors
        """
        neighbor_dict = {}
        if d not in self.row_starts: #left neighbor
            neighbor_dict['left'] = d-1
        if d not in self.row_sums:   #right neighbor
            neighbor_dict['right'] = d+1
        if d not in self.top_vals: #up neighbor
            neighbor_dict['up'] = d + self.up_shift(d)
        if d not in self.base_vals: #down neighbor
            neighbor_dict['down'] = d - self.down_shift(d)
        return neighbor_dict

    def coord(self,d):
        """
        Get the x,y coordinate of a grid location.
        Note: x,y start at 1 as in grid map images
        """
        y = self.get_row(d)
        pos_in_row = d
        if y > 1:
            pos_in_row = d - self.row_sums[y-2]
        x = 8 - (self.row_lengths[y-1]+1)//2 + pos_in_row
        return x,y

    def grid_num(self,x,y):
        """
        Get grid location from x,y coordinate.
        Note: x,y start at 1 as in grid map images
        """
        pos_in_row = x + (self.row_lengths[y-1]+1)//2 - 8
        if y == 1:
            return pos_in_row
        return pos_in_row + self.row_sums[y-2]

    def data_at(self,x,y,inRange=False):
        """
        Get the data at a specific grid x,y
        Note: x,y start at 1 as in grid map images
        """
        if not self.in_grid(x,y):
            return None
        if inRange:
            return self.data[self.grid_num(x,y)][self.min_index:self.max_index,:]
        return self.data[self.grid_num(x,y)]

    def data_at_loc(self,d,inRange=False):
        """
        Get the data at a specific grid Locations
        """
        x,y = self.coord(d)
        return self.data_at(x,y,inRange)

    def get_data_array(self):
        """
        Get all of the measurement data as array
        shape = (# of grid locations, length of each spectra)

        Spectra are converted to 1D height arrays
        """
        data = np.empty(shape=(self.size,self.data_length))
        for i in range(self.size):
            data[i] = self.data[i+1][:,1]
        print(np.shape(data))
        return data

    def in_grid(self,x,y):
        """
        Verify that a grid x,y coordinate is in the grid
        Note: x,y start at 1 as in grid map images
        """
        if x > 15 or x <= 0:
            return False
        if y > 15 or y <= 0:
            return False
        if y == 1 or y == 15:
            if x < 6 or x > 10:
                return False
        if y == 2 or y == 14:
            if x < 4 or x > 12:
                return False
        if y == 3 or y == 13:
            if x < 3 or x > 13:
                return False
        if y in [4,5,11,12]:
            if x < 2 or x > 14:
                return False
        return True
