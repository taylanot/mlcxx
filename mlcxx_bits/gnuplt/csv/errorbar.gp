# Define the filenames
if (!exists("files")) {
    print "Error: filename variable is not set. Please set it before running the script."
    exit 1
}

dark2_colors = "#1b9e77 #d95f02 #7570b3 #e7298a #66a61e #e6ab02 #a6761d #666666"


# Set up the plot
set datafile separator ","

# Set titles and labels
set title "Multiple Data Files Plot"
set xlabel "Column-1"
set ylabel "Column-2"

# Set style for the plot

do for [i=1:words(dark2_colors)] {
    set style line i lt 1 lc rgb word(dark2_colors, i) lw 2 pt 7 ps 1.5
}

# Initialize the plot command

plot for [i=1:words(files)] word(files, i) using 1:2:3 with errorlines linestyle i title word(files, i)


pause mouse close
