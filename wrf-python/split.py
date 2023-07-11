import glob
import subprocess
import os
import multiprocessing as mp
import sys
import shutil

def concat(args):
	path,name=args
    # Specify the directory path containing the wrfout files

	variable_list = "SNOWH,RAINNC,T2,P,HGT"

	sorted_files = sorted(glob.glob(path + '/wrfout_d02*'))

	file_name = name + '_concate.nc'
	try:
       		 # Concatenate the sorted files using ncecat
		subprocess.run(["ncrcat"] + sorted_files + ["-O", file_name])
	except Exception as e:
		print(e)

	if variable_list:
		file_name2 = name + "_sliced.nc"


        	# subset the data
		command = f"ncks -v {variable_list} {file_name} -O {file_name2}"
        
       		 # Execute the command
        
		try:
			subprocess.run(command, shell=True)
            
		except Exception as e:
			print(e)

	os.remove(file_name)

    # Move the files to the right directory
	shutil.move(file_name2, dir)


if __name__ == "__main__":
	dir = '/bsuhome/stanleyakor/WRF-Analysis/schemes/'
	dir_thomp = dir +'thompson-8'
	dir_wsm5 = dir + 'wsm5-4'
	dir_wsm6 = dir + 'wsm6-6'
	dir_milb8 = dir + 'milbrandt-9'

	DIR = [dir_thomp,dir_wsm5,dir_wsm6,dir_milb8]
	Name = ["thompson", "wsm5", "wsm6" , "milbrandt"]

	with mp.Pool(processes=int(sys.argv[1])) as pool:
		results_async = pool.map_async(concat, zip(DIR,Name))

		pool.close()
		pool.join()

		results = results_async.get()
