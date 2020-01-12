#define MAX_CLASS 13
#define NUM_FEATURES 12
#define MAX_DIFF 40 /*MAX_DIFFERENCE*/
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

/****************harris******************/
#define th_harris 0.0001
#define lambda 0.04
/**************************************/

/*英数字一文字（符号なし）（unsigned char）*/
typedef unsigned char img_t;

/*この構造体にはよく代入されるよ笑*/
/*今後出てくるIMAGEや*IMAGEはこの構造体struct IMAGE型を示している*/
typedef struct {
  int height;
  int width;
  int maxval;
  img_t **val;
} IMAGE;

/*******************Harris******************/
typedef struct
{
  int **bool_h;
} HARRIS;
/*******************************************/

/*関数の宣言*/
void *alloc_mem(size_t);
void **alloc_2d_array(int, int, int);
/*struct IMAGE型の関数ポインタを宣言している？？？*/
IMAGE *alloc_image(int, int, int);
void free_image(IMAGE *);
IMAGE *read_pgm(char *);
void write_pgm(IMAGE *, char *);
double calc_snr(IMAGE *, IMAGE *);
void set_thresholds(IMAGE **, IMAGE **, int, int, double *);
int get_label(IMAGE *, IMAGE *, int, int, int, double *);
int get_fvector(IMAGE *, int, int, double, double *);
double cpu_time(void);

/*****************************************harris_directoin**********************************************************/
int harris_direction(IMAGE *, int, int);
int get_fvector_harris(IMAGE *, int, int, double, double *, int);
/******************************************************************************************************************/

/******************************************************************************************************************/
/******************************************************************************************************************/
/******************************************************************************************************************/
/*                                                                                                                */
/*                                                                                                                */
/*                                                                                                                */
/*                                             Harris corner detection                                            */
//harris.c
HARRIS *alloc_harris(int, int);
IMAGE *lowpassGauss_org(IMAGE *, double *, int, int, int);
double **convolve(IMAGE *, int *, int, int);
double **convolve_Gauss(double **, double *, int, int);
double **square(double **, int, int);
double **dxdy_calc(double **, double **, int, int);
double **harris_calc(double **, double **, double **, int, int);
void harris_feature(HARRIS **, HARRIS *, double **, int, int, int);
void set_harris(HARRIS *, HARRIS **, IMAGE **, int);
//pfsvm_common.c
void set_thresholds_harris(IMAGE **, IMAGE **, int, int, double *, double *, HARRIS **);
//harris_for_check
void harris_feature_for_check(HARRIS **, HARRIS *, HARRIS *, double **, int, int, int);
void set_harris_for_check(HARRIS *, HARRIS **, IMAGE **, int);
void write_harris_yuv(IMAGE *, HARRIS *, char *, int, int, int);
/*                                                                                                                */
/*                                                                                                                */
/******************************************************************************************************************/
/******************************************************************************************************************/
/******************************************************************************************************************/