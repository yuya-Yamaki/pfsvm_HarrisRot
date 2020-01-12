#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <dirent.h>
#include <limits.h>
#include "svm.h"
#include "pfsvm.h"
struct svm_parameter param;
struct svm_problem prob;
struct svm_model *model;
struct svm_node *x_space;
//blk_information*************************
struct svm_parameter param_blk;
struct svm_problem prob_blk;
struct svm_model *model_blk;
struct svm_node *x_space_blk;
//***************************************
#define LEAVE_ONE_OUT
#define RND_SEED 12345L
#ifdef LEAVE_ONE_OUT
#define MAX_IMAGE 256
#define SAMPLE_RATIO 0.01
#else
#define MAX_IMAGE 1
#define SAMPLE_RATIO 1.01
#endif

int set_images(char *org_dir, char *dec_dir, IMAGE **oimg_list, IMAGE **dimg_list)
{
#ifdef LEAVE_ONE_OUT
	FILE *fp;
	DIR *dir;
	struct dirent *dp;
	char org_img[256], dec_img[256];
	int num_img = 0;

	if ((dir = opendir(org_dir)) == NULL)
	{
		fprintf(stderr, "Can't open directory '%s'\n", org_dir);
		exit(1);
	}
	while ((dp = readdir(dir)) != NULL)
	{
		if (strncmp(dp->d_name + strlen(dp->d_name) - 4, ".pgm", 4) != 0)
			continue;
		strncpy(org_img, org_dir, 255);
		if (org_img[strlen(org_img) - 1] != '/')
		{
			strcat(org_img, "/");
		}
		strcat(org_img, dp->d_name);
		strncpy(dec_img, dec_dir, 255);
		if (dec_img[strlen(dec_img) - 1] != '/')
		{
			strcat(dec_img, "/");
		}
		strcat(dec_img, dp->d_name);
		strcpy(dec_img + strlen(dec_img) - 4, "-dec.pgm");
		if ((fp = fopen(dec_img, "r")) == NULL)
			continue;
		fclose(fp);
		printf("%s %s\n", org_img, dec_img);
		oimg_list[num_img] = read_pgm(org_img);
		dimg_list[num_img] = read_pgm(dec_img);
		num_img++;
	}
	return (num_img);
#else
	oimg_list[0] = read_pgm(org_dir);
	dimg_list[0] = read_pgm(dec_dir);
	return (1);
#endif
}

int main(int argc, char **argv)
{
	IMAGE *oimg_list[MAX_IMAGE], *dimg_list[MAX_IMAGE], *org, *dec;
	int cls[MAX_CLASS];
	int i, j, k, m, n, label, img;
	int num_img, num_class = 3;
	size_t elements;
	double th_list[MAX_CLASS / 2], fvector[NUM_FEATURES], sig_gain = 1.0;
	const char *error_msg;
	static double svm_c = 1.0, svm_gamma = 1.0 / NUM_FEATURES;
	static char *org_dir = NULL, *dec_dir = NULL, *modelfile = NULL;

	/****************************blk information****************************/
	static char *modelfile_blk = NULL;
	double th_list_blk[MAX_CLASS / 2], fvector_blk[NUM_FEATURES];
	size_t elements_blk;
	int cls_blk[MAX_CLASS];
	int s, t;
	const char *error_msg_blk;
	char tmp[1];
	FILE *TUinfo;
	int x, y, w, h;
	int cux = 0, cuy = 0;
	int bx = 0, by = 0;
	char filename[100];
	int direction = 0;
	int blkcorner_x = 0, blkcorner_y = 0, blkcorner = 0;
	/***********************************************************************/

	/*************************harris direction****************/
	HARRIS *harris;
	HARRIS *harris_list[MAX_IMAGE];
	/********************************************************/

	cpu_time();
	setbuf(stdout, 0);
	for (i = 1; i < argc; i++)
	{
		if (argv[i][0] == '-')
		{
			switch (argv[i][1])
			{
			case 'L':
				num_class = atoi(argv[++i]);
				if (num_class < 3 || num_class > MAX_CLASS || (num_class % 2) == 0)
				{
					fprintf(stderr, "# of classes is wrong!\n");
					exit(1);
				}
				break;
			case 'C':
				svm_c = atof(argv[++i]);
				break;
			case 'G':
				svm_gamma = atof(argv[++i]);
				break;
			case 'S':
				sig_gain = atof(argv[++i]);
				break;
			default:
				fprintf(stderr, "Unknown option: %s!\n", argv[i]);
				exit(1);
			}
		}
		else
		{
			if (org_dir == NULL)
			{
				org_dir = argv[i];
			}
			else if (dec_dir == NULL)
			{
				dec_dir = argv[i];
			}
			else if (modelfile == NULL)
			{
				modelfile = argv[i];
			}
			else
			{
				modelfile_blk = argv[i];
			}
		}
	}
	if (modelfile == NULL || modelfile_blk == NULL)
	{
#ifdef LEAVE_ONE_OUT
		printf("Usage: %s [options] original_dir decoded_dir model.svm model_blk.svm\n",
#else
		printf("Usage: %s [options] original.pgm decoded.pgm model.svm model_blk.svm\n",
#endif
			   argv[0]);
		printf("    -L num  The number of classes [%d]\n", num_class);
		printf("    -C num  Penalty parameter for SVM [%f]\n", svm_c);
		printf("    -G num  Gamma parameter for SVM [%f]\n", svm_gamma);
		printf("    -S num  Gain factor for sigmoid function [%f]\n", sig_gain);
		exit(0);
	}

	num_img = set_images(org_dir, dec_dir, oimg_list, dimg_list);

	/*******************Harris******************/
	harris = (HARRIS *)calloc(1, sizeof(HARRIS));
	set_harris(harris, harris_list, oimg_list, num_img);
	//set_harris_for_check(harris, harris_list, oimg_list, num_img);
	/*******************************************/

	set_thresholds_blk(oimg_list, dimg_list, num_img, num_class, th_list, th_list_blk);
	printf("Number of classes = %d\n", num_class);
	printf("Number of training images = %d\n", num_img);
	printf("Thresholds = {%.1f", th_list[0]);
	for (k = 1; k < num_class / 2; k++)
	{
		printf(", %.1f", th_list[k]);
	}
	printf("}\n");
	printf("Thresholds blk = {%.1f", th_list_blk[0]);
	for (k = 1; k < num_class / 2; k++)
	{
		printf(", %.1f", th_list_blk[k]);
	}
	printf("}\n");
	printf("Gain factor = %f\n", sig_gain);
	printf("SVM(gamma, C) = (%f,%f)\n", svm_gamma, svm_c);

	elements = 0;
	elements_blk = 0;
	prob.l = 0;
	prob_blk.l = 0;
	srand48(RND_SEED); //drand48()のための初期化
	for (img = 0; img < num_img; img++)
	{
		org = oimg_list[img];
		dec = dimg_list[img];
		sprintf(filename, "TUinfo%d.log", img);
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
						if (drand48() < SAMPLE_RATIO)
						{
							if (((j != bx + w - 1) && (j != bx) && (i != by + h - 1) && (i != by)) || (j == 0 && i == 0) || (j == org->width - 1 && i == 0) || (j == 0 && i == org->height - 1) || (j == org->width - 1 && i == org->height - 1) || (j == 0 && i != by + h - 1 && i != by) || (i == 0 && j != bx + w - 1 && j != bx) || (j == org->width - 1 && i != by + h - 1 && i != by) || (i == org->height - 1 && j != bx + w - 1 && j != bx))
							{
								//in blk
								elements += get_fvector(dec, i, j, sig_gain, fvector);
								prob.l++;
							}
							else if ((j == bx + w - 1 && j != org->width - 1) || (j == bx && j != 0) || (i == by + h - 1 && i != org->height - 1) || (i == by && i != 0))
							{
								//blk boundary
								elements_blk += get_fvector(dec, i, j, sig_gain, fvector_blk);
								prob_blk.l++;
							}
						}
					}
				}
			}
		} //while
		fclose(TUinfo);
	} //img
	printf("Number of samples in blk = %d (%d)\n", prob.l, (int)elements);
	printf("Numver of samples blk boundary = %d (%d)\n", prob_blk.l, (int)elements_blk);

	/* Setting for LIBSVM */
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;		 /* for poly */
	param.gamma = svm_gamma; /* for poly/rbf/sigmoid */
	param.coef0 = 0;		 /* for poly/sigmoid */

	/* these are for training only */
	param.nu = 0.5;			   /* for NU_SVC, ONE_CLASS, and NU_SVR */
	param.cache_size = 100;	/* in MB */
	param.C = svm_c;		   /* for C_SVC, EPSILON_SVR and NU_SVR */
	param.eps = 1e-3;		   /* stopping criteria */
	param.p = 0.1;			   /* for EPSILON_SVR */
	param.shrinking = 0;	   // Changed /* use the shrinking heuristics */
	param.probability = 0;	 /* do probability estimates */
	param.nr_weight = 0;	   /* for C_SVC */
	param.weight_label = NULL; /* for C_SVC */
	param.weight = NULL;	   /* for C_SVC */
	elements += prob.l;
	prob.y = Malloc(double, prob.l);
	prob.x = Malloc(struct svm_node *, prob.l);
	x_space = Malloc(struct svm_node, elements);

	/****************************blk information****************************/
	/* Setting for LIBSVM */
	param_blk.svm_type = C_SVC;
	param_blk.kernel_type = RBF;
	param_blk.degree = 3;		 /* for poly */
	param_blk.gamma = svm_gamma; /* for poly/rbf/sigmoid */
	param_blk.coef0 = 0;		 /* for poly/sigmoid */

	/* these are for training only */
	param_blk.nu = 0.5;			   /* for NU_SVC, ONE_CLASS, and NU_SVR */
	param_blk.cache_size = 100;	/* in MB */
	param_blk.C = svm_c;		   /* for C_SVC, EPSILON_SVR and NU_SVR */
	param_blk.eps = 1e-3;		   /* stopping criteria */
	param_blk.p = 0.1;			   /* for EPSILON_SVR */
	param_blk.shrinking = 0;	   // Changed /* use the shrinking heuristics */
	param_blk.probability = 0;	 /* do probability estimates */
	param_blk.nr_weight = 0;	   /* for C_SVC */
	param_blk.weight_label = NULL; /* for C_SVC */
	param_blk.weight = NULL;	   /* for C_SVC */
	elements_blk += prob_blk.l;
	prob_blk.y = Malloc(double, prob_blk.l);
	prob_blk.x = Malloc(struct svm_node *, prob_blk.l);
	x_space_blk = Malloc(struct svm_node, elements_blk);

	/***********************************************************************/

	for (k = 0; k < num_class; k++)
	{
		cls[k] = 0;
		cls_blk[k] = 0;
	}

	m = n = 0;
	s = t = 0;
	srand48(RND_SEED); //drand48のためにsrand48関数で初期化
	for (img = 0; img < num_img; img++)
	{
		org = oimg_list[img];
		dec = dimg_list[img];
		harris = harris_list[img];

		sprintf(filename, "TUinfo%d.log", img);
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
				for (i = by; i < by + h; i++)
				{
					for (j = bx; j < bx + w; j++)
					{
						if (drand48() < SAMPLE_RATIO)
						{
							if (((j != bx + w - 1) && (j != bx) && (i != by + h - 1) && (i != by)) || (j == 0 && i == 0) || (j == org->width - 1 && i == 0) || (j == 0 && i == org->height - 1) || (j == org->width - 1 && i == org->height - 1) || (j == 0 && i != by + h - 1 && i != by) || (i == 0 && j != bx + w - 1 && j != bx) || (j == org->width - 1 && i != by + h - 1 && i != by) || (i == org->height - 1 && j != bx + w - 1 && j != bx))
							{
								//in blk
								label = get_label(org, dec, i, j, num_class, th_list);
								cls[label]++;
								prob.y[m] = label;
								prob.x[m] = &x_space[n];

								//get_fvector(dec, i, j, sig_gain, fvector);
								if (harris->bool_h[i][j] == 1)
								{
									direction = harris_direction(dec, i, j);
								}
								else
								{
									direction = 0;
								}
								get_fvector_direction(dec, i, j, sig_gain, fvector, direction);

								for (k = 0; k < NUM_FEATURES; k++)
								{
									if (fvector[k] != 0.0)
									{
										x_space[n].index = k + 1;
										x_space[n].value = fvector[k];
										n++;
									}
								}
								x_space[n++].index = -1;
								m++;
							}

							else if ((j == bx + w - 1 && j != org->width - 1) || (j == bx && j != 0) || (i == by + h - 1 && i != org->height - 1) || (i == by && i != 0))
							{
								//blk boundary
								//ブロックの隅であるとき
								if ((j == bx && i == by && j != 0 && i != 0) || (j == bx && i == by + h - 1 && j != 0 && i != org->height - 1) || (j == bx + w - 1 && i == by && j != org->width - 1 && i != 0) || (j == bx + w - 1 && i == by + h - 1 && j != org->width - 1 && i != org->height - 1))
								{
									blkcorner_x = j % 4;
									blkcorner_y = i % 4;
									switch (blkcorner_x)
									{
									case 0:
										if (blkcorner_y == 0)
										{
											blkcorner = 1;
										} //左上
										else
										{
											blkcorner = 3;
										} //左下
										break;
									case 3:
										if (blkcorner_y == 0)
										{
											blkcorner = 2;
										} //右上
										else
										{
											blkcorner = 4;
										} //右下
										break;
									}
									direction = slope(org, i, j, blkcorner);
								}

								//境界線方向：horizon上
								else if ((i == by && i != 0 && j != bx && j != bx + w - 1) || (i == by && i != 0 && j == 0) || (i == by && i != 0 && j == org->width - 1))
								{
									direction = 0;
								}

								//境界線方向：horizon下
								else if ((i == by + h - 1 && i != org->height - 1 && j != bx && j != bx + w - 1) //yokosita
										 || (i == by + h - 1 && i != org->height - 1 && j == 0) || (i == by + h - 1 && i != org->height - 1 && j == org->width - 1))
								{
									direction = 2;
								}

								//境界線方向：vertical右
								else if ((j == bx + w - 1 && j != org->width - 1 && i != by && i != by + h - 1) //tatemigi
										 || (j == bx + w - 1 && j != org->width - 1 && i == 0) || (j == bx + w - 1 && j != org->width - 1 && i == org->height - 1))
								{
									direction = 1;
								}

								//境界線方向：vertical左
								else if ((j == bx && j != 0 && i != by && i != by + h - 1) //tatehidari
										 || (j == bx && j != 0 && i == 0) || (j == bx && j != 0 && i == org->height - 1))
								{
									direction = 3;
								}

								label = get_label(org, dec, i, j, num_class, th_list_blk);
								cls_blk[label]++;
								prob_blk.y[s] = label;
								prob_blk.x[s] = &x_space_blk[t];
								get_fvector_blk(dec, i, j, sig_gain, fvector_blk, direction);
								for (k = 0; k < NUM_FEATURES; k++)
								{
									if (fvector_blk[k] != 0.0)
									{
										x_space_blk[t].index = k + 1;
										x_space_blk[t].value = fvector_blk[k];
										t++;
									}
								}
								x_space_blk[t++].index = -1;
								s++;
							}
						} // if drand
					}
				}
			}
		} //while
		fclose(TUinfo);
	} //for img

	for (k = 0; k < num_class; k++)
	{
		printf("CLASS[%d] = %d\n", k, cls[k]);
	}
	for (k = 0; k < num_class; k++)
	{
		printf("CLASS[%d] = %d\n", k, cls_blk[k]);
	}
	error_msg = svm_check_parameter(&prob, &param);
	error_msg_blk = svm_check_parameter(&prob_blk, &param_blk);
	if (error_msg || error_msg_blk)
	{
		fprintf(stderr, "ERROR: %s\n", error_msg);
		fprintf(stderr, "ERROR: %s\n", error_msg_blk);
		exit(1);
	}
	model = svm_train(&prob, &param);
	model_blk = svm_train(&prob_blk, &param_blk);
	if (svm_save_model(modelfile, model) || svm_save_model(modelfile_blk, model_blk))
	{
		fprintf(stderr, "Can't save model to file %s %s\n", modelfile, modelfile_blk);
		exit(1);
	}
	svm_free_and_destroy_model(&model);
	svm_free_and_destroy_model(&model_blk);
	svm_destroy_param(&param);
	svm_destroy_param(&param_blk);
	free(prob.y);
	free(prob_blk.y);
	free(prob.x);
	free(prob_blk.x);
	free(x_space);
	free(x_space_blk);
	printf("cpu time: %.2f sec.\n", cpu_time());
	return (0);
}
