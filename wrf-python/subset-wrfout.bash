#!/bin/bash

# Define the function serial_subset
serial_subset() {
    # Get the function arguments
    path=$1
    variable_list=$2
    output_filename=$3
    outputfile_path=$4

    # Specify the directory path containing the wrfout files
    sorted_files=$(ls -1 "$path/wrfout_d02"* | sort)

    outfile='_concate.nc'
    # Concatenate the sorted files using ncecat
    ncrcat "${sorted_files[@]}" -O "$outfile" || {
        echo "Error in concatenating the files."
        return 1
    }

    # Subset the data
    command="ncks -v $variable_list $outfile -O $output_filename"
    # Execute the command
    eval $command || {
        echo "Error in subsetting the data."
        rm "$outfile"
        return 1
    }

    rm "$outfile"

    # Move the files to the right directory
    if [ ! -d "$outputfile_path" ]; then
        mkdir -p "$outputfile_path" || {
            echo "Error creating the output directory."
            return 1
        }
    fi

    if [ "$condition" = false ]; then
        mv "$output_filename" "$outputfile_path"
    fi

    if [ "$condition" = true ]; then
        command="ncra --mro -d Time,,,24,24 '$output_filename' 'daily_mean_$output_filename'"
        eval $command || {
            echo "Error in computing daily mean."
            rm "$output_filename"
            return 1
        }

        rm "$output_filename"
        mv "daily_mean_$output_filename" "$outputfile_path"
    fi
}

condition=false
# Call the function with appropriate arguments
serial_subset "/bsuhome/stanleyakor/WRF-Analysis/schemes/thompson-8" "I_RAINNC,SNOWH,RAINNC,T2,SWDOWN,SWNORM,U10,V10" "Concatenated_thompson.nc" "/bsuhome/stanleyakor/scratch/thompson/"

