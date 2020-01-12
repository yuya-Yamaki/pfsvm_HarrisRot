#include<stdio.h>
#include<stdlib.h>

int main(int argc, char* argv[])
{
  FILE* fp1, * fp2;
  char tmp[256];
  int x,y,w,h;
  int i = atoi(argv[1]);//文字列で表現された数値をint型の数値に変換する
  char filename[100];

  sprintf(filename,"TUinfo%d.log",i);//filenameにTUinfo(num).logと書き込む

  fp1 = fopen("TUinfo.log","rb");
  fp2 = fopen(filename,"wb");

  while(fscanf(fp1,"%s%d%d%d%d",tmp,&x,&y,&w,&h) != EOF) {
    fprintf(fp2,"%s %d %d %d %d\n",tmp,x,y,w,h);
  }

  fclose(fp1);
  fclose(fp2);

  return 0;
}
