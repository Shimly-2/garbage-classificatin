#Example code to test capabilities of the VL53L0X Library
#Use with caution as this is a work in progress

import sys, time
from VL53L0X import VL53L0X


i2c_address = 0x70

sensor = VL53L0X(i2c_address)
sensor.get_id()
print("Sensor ID is :",hex(sensor.idModel),"\n")
print("Sensor Rev is :",hex(sensor.idRev),"\n")

