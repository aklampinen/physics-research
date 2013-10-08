"""
This script imports the data from a STEREO data file given as the first argument, and then plots important radio events from it.
"""

import h5py
import numpy
import matplotlib.pyplot as plt
import sys

##constants ##
delta_t = 0 #Time between subsequent frequency sweeps, calculated below from time array

##Event finding
sig_cutoff = 0 #Cutoff for when a signal is significant enough to be considered an event, set in terms of max_power below
proximity_cutoff = 8 #How many samples apart signals can be and still be considered part of the same event.

##Event matching
stereo_AB_dist = 16*60 #Distance of stereo spacecraft/speed of light. TODO: Calculate this. For now we know it will always be less than two earth orbital radii, hence 16 lightminutes.
time_tolerance = 77.6 #Tolerance for event length differences, in seconds. Currently approximately 2 delta_t.
worthwhile_length = 20 #Length (in time points) an event has to be before we bother plotting it. Too short an event is worthless, it cannot give us enough points to match or find direction accurately. This number times delta_t gives us the minimum time-length of events we will consider.



##end constants


##Open File
if(len(sys.argv) == 1):
    print "Usage: data.py filename"
    exit(2)

filename = sys.argv[1]

try:
    f = h5py.File(filename,"r")
except IOError:
    print "Error opening file, did you mistype the name?"
    exit(1)
##    

##Import data arrays
time_A = numpy.array(f['spectrogram/Time_A'])
time_B = numpy.array(f['spectrogram/Time_B'])
freq_A = numpy.array(f['spectrogram/Frequency_A'][:,0])
freq_B = numpy.array(f['spectrogram/Frequency_B'][:,0])
power_A = numpy.array(f['spectrogram/HFR_Ch1_S1Power_A'])
power_B = numpy.array(f['spectrogram/HFR_Ch1_S1Power_B'])
##

##Fix ugly data by removing rows of nan##
temp_first = numpy.where(numpy.isnan(freq_A))[0][0]
temp_last = numpy.where(numpy.isnan(freq_A))[-1][-1]
freq_A = numpy.delete(freq_A,numpy.s_[temp_first:(temp_last+1)],0)
power_A = numpy.delete(power_A,numpy.s_[temp_first:(temp_last+1)],0)
temp_first = numpy.where(numpy.isnan(freq_B))[0][0]
temp_last = numpy.where(numpy.isnan(freq_B))[-1][-1]
freq_B = numpy.delete(freq_B,numpy.s_[temp_first:(temp_last+1)],0)
power_B = numpy.delete(power_B,numpy.s_[temp_first:(temp_last+1)],0)

if(numpy.where(numpy.isnan(freq_A))[0].size != 0):
     #This is for the earth communication noise on even-numbered days
     temp_first = numpy.where(numpy.isnan(power_A))[1][0]
     temp_last = numpy.where(numpy.isnan(power_A))[1][-1]
     time_A = numpy.delete(time_A,numpy.s_[temp_first:(temp_last+1)],0)
     power_A = numpy.delete(power_A,numpy.s_[temp_first:(temp_last+1)],1)
     time_B = numpy.delete(time_B,numpy.s_[temp_first:(temp_last+1)],0)
     power_B = numpy.delete(power_B,numpy.s_[temp_first:(temp_last+1)],1)
##

#print numpy.where(numpy.isnan(power_A))
#print numpy.where(numpy.isnan(power_B))

##Get statistics which will be useful later##
delta_t = time_A[1]-time_A[0]
min_power = min(power_A.min(),power_B.min()) #Lowest and
max_power =  max(power_A.max(),power_B.max()) #Highest powers recorded in this time period
array_length = len(power_A[0])
if(len(power_A[0]) != len(power_B[0])):
    print "Power arrays are different lengths... Exiting."
    exit(1)
##

##Blank out very noisy channels##
for row_i in xrange(len(power_A)):
    if(row_i < 5): #The lowest channels on Stereo A suck
       power_A[row_i,:] = min_power*numpy.ones(len(power_A[0]))
    if numpy.median(power_A[row_i,:]) > 0.03 * max_power: #If the row has a high median value, noisy in the beginning, throw them out, along with the rows above and below. The choice of 5% is essentially arbitrary, and based on trial and error.
        power_A[row_i,:] = min_power*numpy.ones(len(power_A[0]))
    #Repeat for B
    if  numpy.median(power_B[row_i,:]) > 0.03 * max_power:
        power_B[row_i,:] = min_power*numpy.ones(len(power_B[0]))        
##

##Find the events##
events_A = [[]] #List to hold event lists
events_B = [[]] #List to hold event lists
in_event = False
sig_cutoff = max_power/5 #Significance cutoff
prox_count = 0 #Counts how far from one point to the next.
for time_i in xrange(len(power_A[0])):
    if max(power_A[:,time_i]) > sig_cutoff:
        prox_count = 0
        if(in_event == True):
            events_A[-1].append(time_i)
        else:
            events_A.append([time_i])
            in_event = True
    else:
        if(in_event == True):
            prox_count += 1
            if(prox_count > proximity_cutoff):
                in_event = False

for time_i in xrange(len(power_B[0])):
    if max(power_B[:,time_i]) > sig_cutoff:
        prox_count = 0
        if(in_event == True):
            events_B[-1].append(time_i)
        else:
            events_B.append([time_i])
            in_event = True
    else:
        if(in_event == True):
            prox_count += 1
            if(prox_count > proximity_cutoff):
                in_event = False

events_A = numpy.delete(events_A,0) #Remove null initial events
events_B = numpy.delete(events_B,0)
##


##Try to match events between the two spacecraft
events_comb = [[]] #To hold events that have been correlated between the spacecraft
for event_A in events_A:
    if(len(event_A) < worthwhile_length): #Disregard very short events, no accuracy
        continue 
    t_start_A = time_A[event_A[0]] 
    t_end_A = time_A[event_A[-1]]
    len_event_A = t_end_A - t_start_A
    mid_time_A = 0.5*(t_start_A + t_end_A)
    for event_B in events_B:
        if(len(event_B) < worthwhile_length): #Disregard very short events, no accuracy
            continue 
        t_start_B = time_B[event_B[0]] 
        t_end_B = time_B[event_B[-1]]
        len_event_B = t_end_A - t_start_A
        mid_time_B = 0.5*(t_start_A + t_end_A)
        if(abs(mid_time_B - mid_time_A) > stereo_AB_dist): #If there was more elapsed time than is possible given the distance between the satellites, disregard
            if(t_start_B > t_start_A): #Later events will be even further from correct.
                break
            else:
                continue
        #Check whether events are similar in length, if so, combine
        if(abs(len_event_B-len_event_A) < time_tolerance):
            len_event = max(event_A[-1]-event_A[0],event_B[-1]-event_B[0]) #Take actual length to be that of the longer event (lengths must be equal for plotting to work)
            i_start_A = event_A[0]
            i_start_B = event_B[0]
            events_comb.append((power_A[:,i_start_A:i_start_A+len_event],time_A[i_start_A:i_start_A+len_event],power_B[:,i_start_B:i_start_B+len_event],time_B[i_start_B:i_start_B+len_event]))
            break

events_comb = numpy.delete(events_comb,0) #Remove null initial event
##

## Plot the events ##
cmap = plt.get_cmap("spectral") #Make it pretty
levels = numpy.array([min_power+(i/50.0)*(max_power-min_power) for i in xrange(50)]) #Set the contour lines for the plot

for event_i in xrange(len(events_comb)):
    print events_comb[event_i]
    print events_comb[event_i][1],events_comb[event_i][3]
    plt.title("Event %i Data from: %s" %(event_i,filename[:-3]))
    plt.subplot(2,1,1)
    plt.contourf(events_comb[event_i][1],freq_A,events_comb[event_i][0],levels=levels,cmap=cmap)
    plt.colorbar()
    plt.ylabel("Stereo A Frequency (MHz)")
    plt.xlabel("Stereo A Time (s)")
    plt.subplot(2,1,2)
    plt.contourf(events_comb[event_i][3],freq_B,events_comb[event_i][2],levels=levels,cmap=cmap)
    plt.colorbar()
    plt.ylabel("Stereo B Frequency (MHz)")
    plt.xlabel("Stereo B Time (s)")
    plt.show()
##

##Close File##
f.close()
##
