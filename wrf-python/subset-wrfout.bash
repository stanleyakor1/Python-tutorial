#!/bin/bash

# Define the function serial_subset
serial_subset() {
    # Get the function arguments
    path=$1
    filename=$2
    variable_list=$3
    output_filename=$4
    outputfile_path=$5

    # Specify the directory path containing the wrfout files
    sorted_files=$(ls -1 "$path/wrfout_d02"* | sort)

    outfile='_concate.nc'
    try {
        # Concatenate the sorted files using ncecat
        ncrcat ${sorted_files[@]} -O $outfile
    } catch {
        echo "Error in concatenating the files."
    }

    # Subset the data
    command="ncks -v $variable_list $outfile -O $output_filename"

    # Execute the command
    try {
        eval $command
    } catch {
        echo "Error in subsetting the data."
    }

    rm $outfile

    # Move the files to the right directory
    if [ ! -d "$outputfile_path" ]; then
        try {
            mkdir -p $outputfile_path
        } catch {
            echo "Error creating the output directory."
        }
    fi

    if [ "$condition" = false ]; then
        mv "$output_filename" "$outputfile_path"
    fi

    if [ "$condition" = true ]; then
        command="ncra --mro -d Time,,,24,24 $output_filename daily_mean_$output_filename"
        try {
            eval $command
        } catch {
            echo "Error in computing daily mean."
        }

        rm "$output_filename"

        mv "daily_mean_$output_filename" "$outputfile_path"
    fi
}

condition=false
# Call the function with appropriate arguments
serial_subset "/path/to/wrfout_files" "filename" "SNOWH,RAINNC,T2,P,HGT" "output_filename.nc" "/output/directory/path"

