# Define the filenames
if (!exists("files")) {
    print "Error: filename variable is not set. Please set it before running the script."
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
dark2_colors = "#1b9e77 #d95f02 #7570b3 #e7298a #66a61e #e6ab02 #a6761d #666666"

# Set titles and labels
set title "Multiple Data Files Plot"
set xlabel "Column-1"
set ylabel "Column-2"

# Set style for the plot
set termoption dashed

do for [i=1:words(dark2_colors)] {
    set style line i lt 1 lc rgb word(dark2_colors, i) lw 2 pt 7 ps 1.5
}
do for [i=1:words(dark2_colors)] {
    set style line i+8 lt 2 lc rgb word(dark2_colors, i) lw 2 pt 7 ps 1.5 dt 3
}



# Initialize the plot command

plot for [i=1:words(files)] word(files, i) binary format='%float64' using 1:4:5 with errorlines linestyle i +8 title word(files, i)."-test"

replot for [i=1:words(files)] word(files, i) binary format='%float64' using 1:2:3 with errorlines linestyle i title word(files, i)."-train"


pause mouse close
