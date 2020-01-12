CC = g++
LIBSVM = libsvm-3.21
CFLAG = -O4 -Wall -I $(LIBSVM)

all: pfsvm_train pfsvm_train_loo pfsvm_eval
#オブジェクトファイルのリンク
#pfsvm_common.oとsvm.oはすべてに共通
pfsvm_train: pfsvm_train.o pfsvm_common.o harris.o $(LIBSVM)/svm.o
	$(CC) $(CFLAG) -lm -o $@ $^

pfsvm_train_loo: pfsvm_train_loo.o pfsvm_common.o harris.o $(LIBSVM)/svm.o
	$(CC) $(CFLAG) -lm -o $@ $^

pfsvm_eval: pfsvm_eval.o pfsvm_common.o harris.o $(LIBSVM)/svm.o
	$(CC) $(CFLAG) -lm -o $@ $^

#コンパイル
pfsvm_common.o: pfsvm_common.c pfsvm.h
pfsvm_train.o: pfsvm_train.c pfsvm.h
pfsvm_train_loo.o: pfsvm_train_loo.c pfsvm.h
pfsvm_eval.o: pfsvm_eval.c pfsvm.h
harris.o: harris.c pfsvm.h
$(LIBSVM)/svm.o:
	cd $(LIBSVM); make

.c.o :
	$(CC) $(CFLAG) -c $<

clean:
	rm -f pfsvm_train pfsvm_train_loo pfsvm_eval harris *.o
