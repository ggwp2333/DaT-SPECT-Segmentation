import os
import sys
import gc
import scipy.io as sio
import numpy as np

def execute(cmd,db_flag):
    if db_flag:
        print(cmd)
    os.system(cmd)

pat_ind_1 = int(sys.argv[1])
pat_ind_2 = int(sys.argv[2])

dict_dir = '' # Specify input directory here
export_dir = '' # Specify output directory here

for pat_ind in (pat_ind_1,pat_ind_2):

    os.chdir(export_dir + '/' + str(pat_ind) + '/')
    
    execute('pwd', 1)
    
    ######################################################################################################################
    ### Prepare SIMIND parameter files
    execute('cp /data02/user-storage/ziping/brain_SPECT_segm/data/params/simind_simul.smc ' + export_dir+'/'+str(pat_ind)+'/simind_simul.smc', 0)
    execute('cp /data02/user-storage/ziping/brain_SPECT_segm/data/params/scattwin.win ' + export_dir+'/'+str(pat_ind)+'/scattwin.win', 0)
    ######################################################################################################################

    ######################################################################################################################
    ### Run SIMIND simulations
    execute('simind simind_simul.smc lc/FS:lc/FD:hrv/NN:1500', 1)     
    execute('rm ' + export_dir+'/'+str(pat_ind)+'/lc_act_av.bin', 1)
        
    execute('simind simind_simul.smc rc/FS:rc/FD:hrv/NN:1500', 1)     
    execute('rm ' + export_dir+'/'+str(pat_ind)+'/rc_act_av.bin', 1)

    execute('simind simind_simul.smc lp/FS:lp/FD:hrv/NN:1000', 1)     
    execute('rm ' + export_dir+'/'+str(pat_ind)+'/lp_act_av.bin', 1)
    
    execute('simind simind_simul.smc rp/FS:rp/FD:hrv/NN:1000', 1)     
    execute('rm ' + export_dir+'/'+str(pat_ind)+'/rp_act_av.bin', 1)
    
    execute('simind simind_simul.smc lgp/FS:lgp/FD:hrv/NN:3500', 1)   #5000
    execute('rm ' + export_dir+'/'+str(pat_ind)+'/lgp_act_av.bin', 1)
    
    execute('simind simind_simul.smc rgp/FS:rgp/FD:hrv/NN:3500', 1)      
    execute('rm ' + export_dir+'/'+str(pat_ind)+'/rgp_act_av.bin', 1)

    execute('simind simind_simul.smc bkg/FS:bkg/FD:hrv/NN:5', 1)      
    execute('rm ' + export_dir+'/'+str(pat_ind)+'/bkg_act_av.bin', 1)