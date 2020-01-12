#! /bin/sh
LOO=./looHD.sh
ORG_DIR=~/HD_pgm/
for IMG in `ls $ORG_DIR*.pgm`
do
	$LOO $IMG
done
