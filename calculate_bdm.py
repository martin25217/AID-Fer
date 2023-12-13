import numpy as np
import struct


def array_to_bitstream(arr):
    # Flatten the array to a 1D array
    flattened_arr = arr.flatten()

    # Convert numbers to their binary representation
    bitstream = ''
    for num in flattened_arr:
        if np.issubdtype(arr.dtype, np.integer):
            bitstream += format(num, 'b')  # For integers
        elif np.issubdtype(arr.dtype, np.floating):
            packed = struct.pack('!f', num)  # For floating-point numbers
            int_representation = struct.unpack('!I', packed)[0]
            bitstream += format(int_representation, 'b')


    return bitstream

def bitstream_to_binarray(bitstream : str, optimized = True):
    strlen = len(bitstream)
    list_of_nums = list(bitstream)

    result = np.empty(strlen, dtype=int)
    counter = 0
    if(optimized):
        for i in list_of_nums:
            if(i == '0'): result[counter] = 0
            if(i == '1'): result[counter] = 1
            counter = counter + 1
    else:
        for i in range(0,len(list_of_nums),2):
            if(list_of_nums[i] == '0' and list_of_nums[i+1] == '0'): result[counter] = 0
            elif(list_of_nums[i] == '0' and list_of_nums[i+1] == '1'): result[counter] = 1
            elif(list_of_nums[i] == '1' and list_of_nums[i+1] == '0'): result[counter] = 2
            elif(list_of_nums[i] == '1' and list_of_nums[i+1] == '1'): result[counter] = 3
            counter = counter + 1
    return result

def calculate_bdm(array, normalized = False, optimization = True):
    from pybdm import BDM
    bdm = BDM(ndim=1)
    if(optimization):
        bitstream = array_to_bitstream(array)
        binarray = bitstream_to_binarray(bitstream)
        if(normalized): return bdm.nbdm(binarray)
        else: return bdm.bdm(binarray, nsymbols = 4)
    else:
        bitstream = array_to_bitstream(array)
        binarray = bitstream_to_binarray(bitstream, optimized=False)
        if(normalized): return bdm.nbdm(binarray)
        else: return bdm.bdm(binarray)


    