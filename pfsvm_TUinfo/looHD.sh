#! /bin/sh
QP=32
SAO=0
LEVEL=3
C=2.0
GAMMA=0.25
GAIN=0.125
count=0
TESTIMAGE=$1
if [ $# -ne 1 ];
then
	echo usage: $0 testimage.pgm
	exit
fi
TESTIMAGE=`basename $TESTIMAGE`

TASK=`basename $TESTIMAGE .pgm`"-q$QP-sao$SAO-c$C-gm$GAMMA-ga$GAIN-l$LEVEL"
MODEL="./model/$TASK.svm"
MODELBLK="./modelblk/$TASK-blk.svm"
LOG="./log/$TASK.log"

HM=~/HM-16.9/
HMOPT="-c $HM""cfg/encoder_intra_main_rext.cfg -cf 400 -f 1 -fr 1 --InternalBitDepth=8"
HMENC="$HM""bin/TAppEncoderStatic"
#HMDEC="$HM""bin/TAppDecoderStatic"
HMDEC="$HM""bin/TAppDecoderAnalyserStatic"
AVSNR=~/avsnr/avsnr
ORG_DIR=~/HD_pgm/
DEC_DIR=./dec_dir

echo "Running at "`uname -a` > $LOG
echo "Test image is $TESTIMAGE" | tee -a $LOG
if [ -d $DEC_DIR ];
then
	rm -r $DEC_DIR
fi
mkdir $DEC_DIR

for ORG_IMG in `ls $ORG_DIR*.pgm`
do
	if [ `basename $ORG_IMG` != $TESTIMAGE ];
	then
		DEC_IMG=`basename $ORG_IMG .pgm`
		DEC_IMG=$DEC_DIR"/$DEC_IMG-dec.pgm"
		echo $ORG_IMG $DEC_IMG
		WIDTH=`pamfile $ORG_IMG | gawk '{print $4}'`
		HEIGHT=`pamfile $ORG_IMG | gawk '{print $6}'`
		tail -n +4 $ORG_IMG > input.y
		$HMENC $HMOPT -q $QP -wdt $WIDTH -hgt $HEIGHT --SAO=$SAO -i input.y
		$HMDEC -d 8 -b str.bin -o rec8bit.y
		rawtopgm $WIDTH $HEIGHT rec8bit.y > $DEC_IMG
   	./import_TUinfo $count
	  echo $count
		count=`echo "$count+1" | bc`
	fi
done
./pfsvm_train_loo -C $C -G $GAMMA -L $LEVEL -S $GAIN $ORG_DIR $DEC_DIR $MODEL $MODELBLK | tee -a $LOG

RATEA=""
SNRA=""
RATEB=""
SNRB=""
ORG_IMG="$ORG_DIR$TESTIMAGE"
WIDTH=`pamfile $ORG_IMG | gawk '{print $4}'`
HEIGHT=`pamfile $ORG_IMG | gawk '{print $6}'`
tail -n +4 $ORG_IMG > input.y
for QP in 37 32 27 22
do
	$HMENC $HMOPT -q $QP -wdt $WIDTH -hgt $HEIGHT --SAO=0 -i input.y
        $HMDEC -d 8 -b str.bin -o rec8bit.y
	rawtopgm $WIDTH $HEIGHT rec8bit.y > reconst.pgm
        SIZE=`ls -l str.bin | gawk '{print $5}'`
        RATEA="$RATEA $SIZE"
        SNR=`pnmpsnr -machine $ORG_IMG reconst.pgm`
        SNRA="$SNRA $SNR"
done

for QP in 37 32 27 22
do
	$HMENC $HMOPT -q $QP -wdt $WIDTH -hgt $HEIGHT --SAO=$SAO -i input.y
        $HMDEC -d 8 -b str.bin -o rec8bit.y
	rawtopgm $WIDTH $HEIGHT rec8bit.y > reconst.pgm
	./pfsvm_eval -S $GAIN $ORG_IMG reconst.pgm $MODEL $MODELBLK modified.pgm | tee -a $LOG
        SIZE=`ls -l str.bin | gawk '{print $5}'`
	SIDE_INFO=`tail $LOG | grep SIDE_INFO | gawk '{print int(($3 + 7) / 8)}'`
	SIZE=`expr $SIZE + $SIDE_INFO`
        RATEB="$RATEB $SIZE"
        SNR=`pnmpsnr -machine $ORG_IMG modified.pgm`
        SNRB="$SNRB $SNR"
done

echo "0" > snr.txt
echo $SNRA | sed -e "s/ /\\t/g" >> snr.txt
echo $RATEA | sed -e "s/ /\\t/g" >> snr.txt
echo $SNRB | sed -e "s/ /\\t/g" >> snr.txt
echo $RATEB | sed -e "s/ /\\t/g" >> snr.txt
cat snr.txt | tee -a $LOG
$AVSNR snr.txt | tee -a $LOG

echo "1" > snr.txt
echo $SNRA | sed -e "s/ /\\t/g" >> snr.txt
echo $RATEA | sed -e "s/ /\\t/g" >> snr.txt
echo $SNRB | sed -e "s/ /\\t/g" >> snr.txt
echo $RATEB | sed -e "s/ /\\t/g" >> snr.txt
cat snr.txt | tee -a $LOG
$AVSNR snr.txt | tee -a $LOG
