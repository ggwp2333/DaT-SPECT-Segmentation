import numpy as np
import os
import pydicom
import sys

def execute(cmd):
    print(cmd)
    os.system(cmd)

pat_ind_1 = int(sys.argv[1])
pat_ind_2 = int(sys.argv[2])

clincal_counts_lvl = 2*10e5

for pat_ind in range(pat_ind_1, pat_ind_2+1):

    dir_n = '' + str(pat_ind)  # Specify input directory
    export_dir_n = dir_n + '/dmip_recon/output'
    
    if os.path.isdir(export_dir_n):
        raise TypeError("Warning: Trying to create a directory that already exists!")
    execute('mkdir ' + export_dir_n)

    execute('imgconv -r ' + dir_n+'/dmip_recon/final_tot_w1.h00 ' + export_dir_n+'/tot_w1.im')
    execute('imgconv -r ' + dir_n+'/dmip_recon/final_tew_w3.h00 ' + export_dir_n+'/tew_w3.im')
    
    sum_tot_w1 = np.sum(np.fromfile(dir_n+'/dmip_recon/final_tot_w1.a00',dtype='float32'))
    
    factor_whole = clincal_counts_lvl/sum_tot_w1
    
    # scale to clinical counts level (primary + scatter)
    # w1 = 4
    # w2 = 4
    # w3 = 175-143 # 32
    # 1/2 (D1/W1 + D2/W2) W3
    tew_factor = 4 #1/2/(w1)*w3 
    
    execute('scale -A ' + str(factor_whole) + ' ' + export_dir_n+'/tot_w1.im ' + export_dir_n+'/tot_w1_clinical_lvl.im')
    execute('scale -A ' + str(factor_whole/tew_factor) + ' ' + export_dir_n+'/tew_w3.im ' + export_dir_n+'/tew_w3_clinical_lvl.im')
    
    atn_file = dir_n+'/hrv_atn_av.bin'
    atn_bin = np.fromfile(atn_file,dtype='float32',sep='')
    atn_np = np.reshape(atn_bin,(512,512,512))*10000
    atn_name = export_dir_n+'/lrv_atn.dcm'
    
    execute('cp /data02/user-storage/ziping/brain_SPECT_segm/data/atn_template.dcm ' + export_dir_n+'/atn_template.dcm')
    
    ds = pydicom.dcmread(export_dir_n+'/atn_template.dcm')
    ds.PixelData = np.array(atn_np,dtype='int16')
    ds.LargestImagePixelValue = np.max(ds.PixelData)
    ds.save_as(atn_name)
    
    execute('imgconv -r ' + atn_name + ' ' + export_dir_n+'/hr_dmip_atn.im')
    execute('interp3d -S 0.25 0.25 0.25 -A 0.0625 ' + export_dir_n+'/hr_dmip_atn.im ' + export_dir_n+'/dmip_atn.im')
    execute('flipim -z -e -x -y '+ export_dir_n+'/dmip_atn.im ' + export_dir_n+'/dmip_atn_final.im')
    
    # One noise realization
    execute('addnoise ' + export_dir_n+'/tot_w1_clinical_lvl.im ' + export_dir_n+'/tot_w1_clinical_lvl_noise.im')
    execute('imgconv ' + export_dir_n+'/tot_w1_clinical_lvl_noise.im ' + export_dir_n+'/tot_w1_clinical_lvl_noise.bin')
    
    execute('addnoise ' + export_dir_n+'/tew_w3_clinical_lvl.im ' + export_dir_n+'/tew_w3_clinical_lvl_noise.im')
    
    # tew scale back
    execute('scale -A ' + str(tew_factor) + ' ' + export_dir_n+'/tew_w3_clinical_lvl_noise.im ' + export_dir_n+'/tew_w3_clinical_lvl_noise_back.im')
    
    # scatter compensation
    execute('add ' + export_dir_n+'/tot_w1_clinical_lvl_noise.im ' + export_dir_n+'/tew_w3_clinical_lvl_noise_back.im ' + export_dir_n+'/dmip_prj.im')
    execute('osems /data02/user-storage/ziping/brain_SPECT_segm/data/params/osem3D_simind_aligned.par ' + export_dir_n+'/dmip_prj.im ' + export_dir_n+'/dmip_atn_final.im ' + export_dir_n+'/dmip_rec')
    
    execute('imgconv -rR ' + export_dir_n+'/dmip_rec.4.im ' + export_dir_n+'/dmip_rec.bin')
    
    execute('rm ' + export_dir_n+'/atn_template.dcm')
    execute('rm ' + export_dir_n+'/lrv_atn.dcm')
    execute('rm ' + export_dir_n+'/hr_dmip_atn.im')
    execute('rm ' + export_dir_n+'/dmip_atn.im')
    
    