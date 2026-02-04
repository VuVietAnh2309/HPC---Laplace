# Heatmap
set terminal png size 800,600
set output 'heatmap.png'
set pm3d map
set xlabel 'X'
set ylabel 'Y'
set title 'Temperature Distribution (Heatmap)'
splot 'sor_solution.dat' using 1:2:3 with pm3d notitle

# Contour
set output 'contour.png'
set contour base
set cntrparam levels 20
set title 'Temperature Distribution (Contour)'
splot 'sor_solution.dat' using 1:2:3 with lines notitle
