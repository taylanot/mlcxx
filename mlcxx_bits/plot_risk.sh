#! /bin/zsh

# gnuplot -e "files='build/risk_estim/outputs/bimodal/5/e.bin build/risk_estim/outputs/bimodal/5/c.bin build/risk_estim/outputs/bimodal/5/t.bin build/risk_estim/outputs/bimodal/5/l.bin'" -e "bins=100" -e "ylog=1" gnuplt/bin/hist.gp

# gnuplot -e "files='build/risk_estim/outputs/lognorm/5/e.bin build/risk_estim/outputs/lognorm/5/c.bin build/risk_estim/outputs/lognorm/5/t.bin build/risk_estim/outputs/lognorm/5/l.bin'" -e "bins=100" -e "ylog=1" gnuplt/bin/hist.gp

 # gnuplot -e "files='build/risk_estim/outputs/pareto/5/e.bin build/risk_estim/outputs/pareto/5/c.bin build/risk_estim/outputs/pareto/5/t.bin build/risk_estim/outputs/pareto/5/l.bin'" -e "bins=100" -e "ylog=1" gnuplt/bin/hist.gp


# gnuplot -e "files='build/risk_estim/outputs/norm/5/e.bin build/risk_estim/outputs/norm/5/c.bin build/risk_estim/outputs/norm/5/t.bin build/risk_estim/outputs/norm/5/l.bin'" -e "bins=100" -e "ylog=1" gnuplt/bin/hist.gp

gnuplot -e "files='build/risk_estim/outputs/pareto/100/e.bin build/risk_estim/outputs/pareto/100/c.bin build/risk_estim/outputs/pareto/100/t.bin build/risk_estim/outputs/pareto/100/l.bin'" -e "bins=100" -e "ylog=1" gnuplt/bin/hist.gp

# gnuplot -e "files='build/risk_estim/outputs2/norm/100/e.bin build/risk_estim/outputs2/norm/100/c.bin build/risk_estim/outputs2/norm/100/t.bin build/risk_estim/outputs2/norm/100/l.bin'" -e "bins=100" -e "ylog=1" gnuplt/bin/hist.gp
