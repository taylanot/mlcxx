# Define the filenames
if (!exists("files")) {
    print "Error: files variable is not set. Please set it before running the script."
    exit 1
}
num_files = 0
do for [i=1:words(files)] {
  num_files = num_files + 1
}

if (!exists("ylog")) {
  ylog = 0
}

if (!exists("xlog")) {
  xlog = 0
}
if (ylog == 1 ) {
    set logscale y 10
}

if (xlog == 1 ) {
    set logscale x 10
}

if (!exists("bins")) {
    print "Error: bins variable is not set. Please set it before running the script."
    exit 1
}


# Set plot title and labels
set title "Histogram Plot"
set xlabel "Value"
set ylabel "Frequency"

dark2_colors = "#1b9e77 #d95f02 #7570b3 #e7298a #66a61e #e6ab02 #a6761d #666666"

# Calculate the data range across all files
overall_min = NaN
overall_max = NaN
array maxs[num_files]
array mins[num_files]
do for [i=1:words(files)] {
    stats word(files, i) using 1 nooutput
    maxs[i] = STATS_max
    mins[i] = STATS_min
    if (i == 1 || STATS_min < overall_min) {
        overall_min = STATS_min
    }
    if (i == 1 || STATS_max > overall_max) {
        overall_max = STATS_max
    }
}
array bin_widths[num_files]
do for [i=1:words(files)] {
  bin_widths[i] = (maxs[i] - mins[i]) / bins
}

#bin(x, width) = floor(x / width) * width
bin(x, width) = width * floor(x / width) + width / 2.0

do for [i=1:words(dark2_colors)] {
    set style line i lt 1 lc rgb word(dark2_colors, i) lw 2 pt 7 ps 1.5
}

set style fill solid

# Initialize the plot command
plot for [i=1:words(files)] word(files, i)  using (bin($1, bin_widths[i])):(1.0) smooth freq with boxes linestyle i title word(files, i)

pause mouse close
