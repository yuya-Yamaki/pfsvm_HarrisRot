#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "pfsvm.h"

/*pfsvm全てに共通するプログラム．makefileをみるとコンパイル時にリンクされているのがわかる．
main関数はない.ここで関数を作ってpfsvm_trainもしくはpfsvm_train_looでここで定義した関数を用いて機械学習を行っている．*/

FILE *fileopen(char *filename, const char *mode)
{
  FILE *fp;
  fp = fopen(filename, mode);
  if (fp == NULL)
  {
    fprintf(stderr, "Can\'t open %s!\n", filename);
    exit(1);
  }
  return (fp);
}

void *alloc_mem(size_t size)
{
  void *ptr;
  if ((ptr = (void *)malloc(size)) == NULL)
  {
    fprintf(stderr, "Can\'t allocate memory (size = %d)!\n", (int)size);
    exit(1);
  }
  return (ptr);
}

void **alloc_2d_array(int height, int width, int size)
{
  void **mat;
  char *ptr;
  int k;

  mat = (void **)alloc_mem(sizeof(void *) * height + height * width * size);
  ptr = (char *)(mat + height);
  for (k = 0; k < height; k++)
  {
    mat[k] = ptr;
    ptr += width * size;
  }
  return (mat);
}

IMAGE *alloc_image(int width, int height, int maxval)
{
  IMAGE *img;
  img = (IMAGE *)alloc_mem(sizeof(IMAGE));
  img->width = width;
  img->height = height;
  img->maxval = maxval;
  img->val = (img_t **)alloc_2d_array(img->height, img->width,sizeof(img_t));
  return (img);
}

void free_image(IMAGE *img)
{
  if (img != NULL && img->val != NULL)
  {
    free(img->val);
    free(img);
  }
  else
  {
    fprintf(stderr, "! error in free_image()\n");
    exit(1);
  }
}

IMAGE *read_pgm(char *filename)
{
  int i, j, width, height, maxval;
  char tmp[256];
  IMAGE *img;
  FILE *fp;

  fp = fileopen(filename, "rb");
  fgets(tmp, 256, fp);
  if (tmp[0] != 'P' || tmp[1] != '5')
  {
    fprintf(stderr, "Not a PGM file!\n");
    exit(1);
  }
  while (*(fgets(tmp, 256, fp)) == '#')
  ;
  sscanf(tmp, "%d %d", &width, &height);
  while (*(fgets(tmp, 256, fp)) == '#')
  ;
  sscanf(tmp, "%d", &maxval);
  img = alloc_image(width, height, maxval);
  for (i = 0; i < img->height; i++)
  {
    for (j = 0; j < img->width; j++)
    {
      img->val[i][j] = (img_t)fgetc(fp);
    }
  }
  fclose(fp);
  return (img);
}

void write_pgm(IMAGE *img, char *filename)
{
  int i, j;
  FILE *fp;
  fp = fileopen(filename, "wb");
  fprintf(fp, "P5\n%d %d\n%d\n", img->width, img->height, img->maxval);
  for (i = 0; i < img->height; i++)
  {
    for (j = 0; j < img->width; j++)
    {
      putc(img->val[i][j], fp);
    }
  }
  fclose(fp);
  return;
}

/*PSNRの計算*/
double calc_snr(IMAGE *img1, IMAGE *img2)
{
  int i, j, d;
  double mse;

  mse = 0.0;
  for (i = 0; i < img1->height; i++)
  {
    for (j = 0; j < img1->width; j++)
    {
      d = img1->val[i][j] - img2->val[i][j];
      mse += d * d;
    }
  }
  mse /= (img1->width * img1->height);
  return (10.0 * log10(255 * 255 / mse));
}

/*thresholds閾値を決定する関数*/
void set_thresholds(IMAGE **oimg_list, IMAGE **dimg_list, int num_img, int num_class, double *th_list)
{
  IMAGE *org, *dec;
  int hist[MAX_DIFF + 1];
  int img, i, j, k;
  double class_size;

  for (k = 0; k < MAX_DIFF + 1; k++)
  {
    hist[k] = 0;
  }
  class_size = 0;
  for (img = 0; img < num_img; img++)
  {
    org = oimg_list[img];
    dec = dimg_list[img];

    /*ラスタスキャン***************************************************************/
    for (i = 0; i < org->height; i++)
    {
      for (j = 0; j < org->width; j++)
      {
        /*k=org->val[i][j](原画像の画素値)-dec->val[i][j](注目画素の再生値)*/
        k = org->val[i][j] - dec->val[i][j];
        if (k < 0)
        k = -k;
        /*eが40を超えたときe=40*/
        if (k > MAX_DIFF)
        k = MAX_DIFF;
        /*e(=原画像の画素値と注目画素の再生値の差)がそれぞれの位置にどれくらい存在しているかを数えている*/
        hist[k]++;
      }
    }
    /***************************************************************ラスタスキャン*/
    class_size += org->width * org->height;
  }
  for (k = 0; k < MAX_DIFF + 1; k++)
  {                                   /*#define MAX_DIFF 40(pfsvm.h)*/
    printf("%4d%8d\n", k, hist[k]); /*この分布から閾値を決定することができる*/
  }
  class_size /= num_class; /*各クラスの生起分布の個数*/
  i = 0;
  for (k = 0; k < MAX_DIFF; k++)
  {
    if (hist[k] > class_size * (1 + 2 * i))
    {
      th_list[i++] = k + 0.5; /************これがクラス分類に用いる閾値************/
    }
    hist[k + 1] += hist[k];
  }
}

/*we use a three-class SVM whitch predicts a class label Y={1,0,-1} from the above feature vector*/
int get_label(IMAGE *org, IMAGE *dec, int i, int j, int num_class, double *th_list) /*num_class=3*/
{
  int d, sgn, k;
  /*d=org->val[i][j](原画像の画素値)-dec->val[i][j](注目画素の再生値)*/
  d = org->val[i][j] - dec->val[i][j];
  if (d > 0)
  {
    sgn = 1;
  }
  else
  {
    d = -d;
    sgn = -1;
  }
  for (k = 0; k < num_class / 2; k++)
  {
    if (d < th_list[k])
    break;
  }
  return (sgn * k + num_class / 2); /*0,1,2でlabelをつけている*/
}

/*特徴ベクトルの取得*/
int get_fvector(IMAGE *img, int i, int j, double gain, double *fvector)
{
  typedef struct
  {
    int y, x;
  } POINT;

  const POINT dyx[] = {
    /* 0 (1) */
    {0, 0},
    /* 1 (5) */
    {0, -1},
    {-1, 0},
    {0, 1},
    {1, 0},
    /* 2 (13) */
    {0, -2},
    {-1, -1},
    {-2, 0},
    {-1, 1},
    {0, 2},
    {1, 1},
    {2, 0},
    {1, -1},
    /* 3 (25) */
    {0, -3},
    {-1, -2},
    {-2, -1},
    {-3, 0},
    {-2, 1},
    {-1, 2},
    {0, 3},
    {1, 2},
    {2, 1},
    {3, 0},
    {2, -1},
    {1, -2}
  };
  int k, x, y, v0, vk, num_nonzero;
  /*注目画素の再生値＝v0*/
  v0 = img->val[i][j];
  num_nonzero = 0;

  /*for構文の繰り返しは12次元特徴ベクトルとしたので12回で良い
  for (k=o; k<12; k++){}
  #define NUM_FEATURES 12;(pfsvm.h)*/
  for (k = 0; k < NUM_FEATURES; k++)
  {
    y = i + dyx[k + 1].y;
    if (y < 0)
    y = 0;
    if (y > img->height - 1)
    y = img->height - 1;
    x = j + dyx[k + 1].x;
    if (x < 0)
    x = 0;
    if (x > img->width - 1)
    x = img->width - 1;
    /*vk=img->val[y][x](周辺画素の再生値)-img->val[i][j](注目画素の再生値)*/
    vk = img->val[y][x] - v0;
    if (vk != 0)
    num_nonzero++;

    /*スケーリング関数（シグモイド関数）用いて-1<特徴ベクトル<+1の範囲にする
    vkが入力（キャスト演算子で小数に変換されている）
    k=0,1,2,...,11の12次元特徴ベクトルが得られる*/
    fvector[k] = 2.0 / (1 + exp(-(double)vk * gain)) - 1.0;
  }
  return (num_nonzero);
}
/*ここまでがget_fvector関数*/

/*処理にかかった時間を出力する*/
double cpu_time(void)
{
  #ifndef CLK_TCK
  #define CLK_TCK 60
  #endif
  static clock_t prev = 0;
  clock_t cur, dif;

  cur = clock();
  if (cur > prev)
  {
    dif = cur - prev;
  }
  else
  {
    dif = (unsigned)cur - prev;
  }
  prev = cur;

  return ((double)dif / CLOCKS_PER_SEC);
}

/***********************************************************************/
/*                                                                     */
/*                                                                     */
/*                           blk information                           */
/*                                                                     */
/*                                                                     */
/***********************************************************************/

void set_thresholds_blk(IMAGE **oimg_list, IMAGE **dimg_list, int num_img, int num_class, double *th_list_inblk, double *th_list_blk_boundary)
{
  IMAGE *org, *dec;
  int hist_inblk[MAX_DIFF + 1];
  int hist_blk_boundary[MAX_DIFF + 1];
  int img, i, j, k;
  double class_size_inblk;
  double class_size_blk_boundary;

  FILE *TUinfo;
  int x, y, w, h;
  int cux = 0, cuy = 0;
  int bx = 0, by = 0;
  char filename[100];
  char tmp[1];

  for (k = 0; k < MAX_DIFF + 1; k++)
  {
    hist_inblk[k] = 0;
    hist_blk_boundary[k] = 0;
  }
  class_size_inblk = 0;
  class_size_blk_boundary = 0;
  for (img = 0; img < num_img; img++)
  {
    org = oimg_list[img];
    dec = dimg_list[img];
    if (num_img == 1)
    {
      sprintf(filename, "TUinfo.log");
    }
    else
    {
      sprintf(filename, "TUinfo%d.log", img);
    }
    TUinfo = fopen(filename, "rb");

    while (fscanf(TUinfo, "%s%d%d%d%d", tmp, &x, &y, &w, &h) != EOF)
    {
      if (tmp[0] == 'C')
      {
        cux = x;
        cuy = y;
      }
      else
      {
        bx = cux + x;
        by = cuy + y;
        //block毎にラスタスキャン
        for (i = by; i < by + h; i++)
        {
          for (j = bx; j < bx + w; j++)
          {
            if (((j != bx + w - 1) && (j != bx) && (i != by + h - 1) && (i != by)) || (j == 0 && i == 0) || (j == org->width - 1 && i == 0) || (j == 0 && i == org->height - 1) || (j == org->width - 1 && i == org->height - 1) || (j == 0 && i != by + h - 1 && i != by) || (i == 0 && j != bx + w - 1 && j != bx) || (j == org->width - 1 && i != by + h - 1 && i != by) || (i == org->height - 1 && j != bx + w - 1 && j != bx))
            {
              //in blk
              k = org->val[i][j] - dec->val[i][j];
              if (k < 0)
              k = -k;
              if (k > MAX_DIFF)
              k = MAX_DIFF;
              hist_inblk[k]++;
              class_size_inblk++;
            }
            else if ((j == bx + w - 1 && j != org->width - 1) || (j == bx && j != 0) || (i == by + h - 1 && i != org->height - 1) || (i == by && i != 0))
            {
              //blk boundary
              k = org->val[i][j] - dec->val[i][j];
              if (k < 0)
              k = -k;
              if (k > MAX_DIFF)
              k = MAX_DIFF;
              hist_blk_boundary[k]++;
              class_size_blk_boundary++;
            }
          }
        }
      }
    }
    fclose(TUinfo);
  }
  for (k = 0; k < MAX_DIFF + 1; k++)
  {
    printf("%4d%8d\n", k, hist_inblk[k]);
  }
  printf("\n");
  for (k = 0; k < MAX_DIFF + 1; k++)
  {
    printf("%4d%8d\n", k, hist_blk_boundary[k]);
  }
  class_size_inblk /= num_class;
  class_size_blk_boundary /= num_class;
  i = 0;
  j = 0;
  for (k = 0; k < MAX_DIFF; k++)
  {
    if (hist_inblk[k] > class_size_inblk * (1 + 2 * i))
    {
      th_list_inblk[i++] = k + 0.5;
    }
    if (hist_blk_boundary[k] > class_size_blk_boundary * (1 + 2 * j))
    {
      th_list_blk_boundary[j++] = k + 0.5;
    }
    hist_inblk[k + 1] += hist_inblk[k];
    hist_blk_boundary[k + 1] += hist_blk_boundary[k];
  }
}

/*特徴ベクトルの取得*/
int get_fvector_blk(IMAGE *img, int i, int j, double gain, double *fvector, int direction)
{
  typedef struct
  {
    int y, x;
  } POINT;

  const POINT dyx[] = {
    /* 0 (1) */
    {0, 0},
    /* 1 (5) */
    {0, -1},
    {-1, 0},
    {0, 1},
    {1, 0},
    /* 2 (13) */
    {0, -2},
    {-1, -1},
    {-2, 0},
    {-1, 1},
    {0, 2},
    {1, 1},
    {2, 0},
    {1, -1}
  };
  int k, x, y, v0, vk, num_nonzero;
  /*注目画素の再生値＝v0*/
  v0 = img->val[i][j];
  num_nonzero = 0;
  int tmp;

  for(k = 0; k < NUM_FEATURES; k++)
  {
    //境界線方向(direction)によって座標を決定できるよう変数化
    tmp = (k + direction) % 4 + 1;
    if(k >= 4){
      tmp = ((k - 4) + (2 * direction)) % 8 + 5;
    }
    y = i + dyx[tmp].y;
    if (y < 0)
    y = 0;
    if (y > img->height - 1)
    y = img->height - 1;
    x = j + dyx[tmp].x;
    if (x < 0)
    x = 0;
    if (x > img->width - 1)
    x = img->width - 1;

    vk = img->val[y][x] - v0;
    if (vk != 0)
    num_nonzero++;

    fvector[k] = 2.0 / (1 + exp(-(double)vk * gain)) - 1.0;
  }
  return (num_nonzero);
}

int slope(IMAGE *img, int i, int j, int blkcorner)
{
  typedef struct{
    int y, x;
  }POINT;

  const POINT dyx[] = {
    { 0, 0},
    {-1, 0}, { 0, 1}, { 1, 0}, { 0,-1}
  };
  int k, x, y, v0;
  int direction = 0;
  int slope[4];
  /*注目画素の再生値＝v0*/
  v0 = img->val[i][j];

  for(k = 0; k < 4; k++){
    y = i + dyx[k + 1].y;
    if(y < 0) y = 0;
    if(y > img->height - 1) y = img->height - 1;
    x = j + dyx[k + 1].x;
    if(x < 0) x = 0;
    if(x > img->width - 1) x = img->width - 1;
    slope[k] = img->val[y][x] - v0;
    if(slope[k] < 0){
      slope[k] = -slope[k];
    }
  }

  switch(blkcorner){
    case 1:
    if(slope[0] > slope[3]) direction = 0;
    else direction = 3;
    break;
    case 2:
    if(slope[0] > slope[1]) direction = 0;
    else direction = 1;
    break;
    case 3:
    if(slope[2] > slope[3]) direction = 2;
    else direction = 3;
    break;
    default:
    if(slope[2] > slope[1]) direction = 2;
    else direction = 1;
    break;
  }
  return direction;
}

/***********************************************************************/
/*                                                                     */
/*                                                                     */
/*                           harris direction                          */
/*                                                                     */
/*                                                                     */
/***********************************************************************/

int harris_direction(IMAGE *dec, int i, int j)
{
  int sobel_x[9] = {
      -1, 0, 1,
      -2, 0, 2,
      -1, 0, 1};

  int sobel_y[9] = {
      -1, -2, -1,
      0, 0, 0,
      1, 2, 1};

  int s, t, x, y;
  int tmp = 0, z = 0;
  int x_direction, y_direction;
  int direction = 0;

  for (s = -1; s <= 1; s++)
  {
    for (t = -1; t <= 1; t++)
    {
      x = j + t;
      if (x < 0)
        x = 0;
      else if (x > dec->width - 1)
        x = dec->width - 1;
      y = i + s;
      if (y < 0)
        y = 0;
      else if (y > dec->height - 1)
        y = dec->height - 1;

      tmp += dec->val[y][x] * sobel_x[z];
      z++;
    }
  }
  x_direction = tmp;
  tmp = 0;
  z = 0;

  for (s = -1; s <= 1; s++)
  {
    for (t = -1; t <= 1; t++)
    {
      x = j + t;
      if (x < 0)
        x = 0;
      else if (x > dec->width - 1)
        x = dec->width - 1;
      y = i + s;
      if (y < 0)
        y = 0;
      else if (y > dec->height - 1)
        y = dec->height - 1;

      tmp += dec->val[y][x] * sobel_y[z];
      z++;
    }
  }
  y_direction = tmp;

  if (abs(x_direction) > abs(y_direction))
  {
    if (x_direction < 0)
      direction = 3;
    else
      direction = 2;
  }
  else if (abs(x_direction) < abs(y_direction))
  {
    if (y_direction < 0)
      direction = 0;
    else
      direction = 1;
  }

  return direction;
}

/*特徴ベクトルの取得*/
int get_fvector_direction(IMAGE *img, int i, int j, double gain, double *fvector, int direction)
{
  typedef struct
  {
    int y, x;
  } POINT;

  const POINT dyx[] = {
      /* 0 (1) */
      {0, 0},
      /* 1 (5) */
      {0, -1},
      {-1, 0},
      {0, 1},
      {1, 0},
      /* 2 (13) */
      {0, -2},
      {-1, -1},
      {-2, 0},
      {-1, 1},
      {0, 2},
      {1, 1},
      {2, 0},
      {1, -1}};
  int k, x, y, v0, vk, num_nonzero;
  /*注目画素の再生値＝v0*/
  v0 = img->val[i][j];
  num_nonzero = 0;
  int tmp;

  for (k = 0; k < NUM_FEATURES; k++)
  {
    //境界線方向(direction)によって座標を決定できるよう変数化
    tmp = (k + direction) % 4 + 1;
    if (k >= 4)
    {
      tmp = ((k - 4) + (2 * direction)) % 8 + 5;
    }
    y = i + dyx[tmp].y;
    if (y < 0)
      y = 0;
    if (y > img->height - 1)
      y = img->height - 1;
    x = j + dyx[tmp].x;
    if (x < 0)
      x = 0;
    if (x > img->width - 1)
      x = img->width - 1;

    vk = img->val[y][x] - v0;
    if (vk != 0)
      num_nonzero++;

    fvector[k] = 2.0 / (1 + exp(-(double)vk * gain)) - 1.0;
  }
  return (num_nonzero);
}