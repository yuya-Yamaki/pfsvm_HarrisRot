#! /bin/sh
LOO=./loo.sh
ORG_DIR=~/HD_pgm/
for IMG in `ls $ORG_DIR*.pgm`
do
	$LOO $IMG
done
