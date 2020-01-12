#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "svm.h"
#include "pfsvm.h"
struct svm_model *model;
struct svm_node *x;

/*pfsvmによって得られた画像を評価するためのプログラム（PSNRetc...）*/

int main(int argc, char **argv)
{
    IMAGE *org, *dec, *cls;
    int i, j, k, n, label, success;
    int num_class, side_info;
    double th_list[MAX_CLASS / 2], fvector[NUM_FEATURES], sig_gain = 1.0;
    double offset[MAX_CLASS];
    int cls_hist[MAX_CLASS];
    double sn_before, sn_after;
    static char *orgimg = NULL, *decimg = NULL, *modelfile = NULL, *modimg = NULL;

    /*************************harris direction****************/
    HARRIS *harris;
    HARRIS *harris_list[1];
    int direction = 0;
    /********************************************************/

    cpu_time();
    setbuf(stdout, 0);
    for (i = 1; i < argc; i++)
    {
        if (argv[i][0] == '-')
        {
            switch (argv[i][1])
            {
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
            if (orgimg == NULL)
            {
                orgimg = argv[i];
            }
            else if (decimg == NULL)
            {
                decimg = argv[i];
            }
            else if (modelfile == NULL)
            {
                modelfile = argv[i];
            }
            else
            {
                modimg = argv[i];
            }
        }
    }
    if (modimg == NULL)
    {
        printf("Usage: %s [option] original.pgm decoded.pgm model.svm modified.pgm\n",
               argv[0]);
        printf("    -S num  Gain factor for sigmoid-like function[%f]\n", sig_gain);
        exit(0);
    }
    /*orgは原画像,decは再生画像*/
    org = read_pgm(orgimg);
    dec = read_pgm(decimg);
    cls = alloc_image(org->width, org->height, 255);
    if ((model = svm_load_model(modelfile)) == 0)
    {
        fprintf(stderr, "can't open model file %s\n", modelfile);
        exit(1);
    }

    /*閾値の取得*/
    num_class = model->nr_class;

    /*******************Harris******************/
    harris = (HARRIS *)calloc(1, sizeof(HARRIS));
    set_harris(harris, harris_list, &dec, 1);
    //set_harris_for_check(harris, harris_list, oimg_list, num_img);
    harris = harris_list[0];
    /*******************************************/

    /*何パーセント正確をはかるためにあるプログラムで実際いらない部分***************************/
    set_thresholds(&org, &dec, 1, num_class, th_list);
    printf("PSNR = %.2f (dB)\n", sn_before = calc_snr(org, dec));
    printf("# of classes = %d\n", num_class);
    printf("Thresholds = {%.1f", th_list[0]);
    for (k = 1; k < num_class / 2; k++)
    {
        printf(", %.1f", th_list[k]);
    }
    printf("}\n");
    printf("Gain factor = %f\n", sig_gain);
    x = Malloc(struct svm_node, NUM_FEATURES + 1);
    success = 0;
    for (k = 0; k < num_class; k++)
    {
        offset[k] = 0.0;
        cls_hist[k] = 0;
    }
    /***************************************************************************************/

    /*特徴ベクトルの取得,SVMの入力となる．この特徴ベクトルを用いてどこに分類されるかを機械学習で予測する*/
    for (i = 0; i < dec->height; i++)
    {
        for (j = 0; j < dec->width; j++)
        {
            //get_fvector(dec, i, j, sig_gain, fvector);
            if (harris->bool_h[i][j] == 1)
            {
                direction = harris_direction(dec, i, j);
            }
            else
            {
                direction = 0;
            }
            get_fvector_harris(dec, i, j, sig_gain, fvector, direction);
            n = 0;
            for (k = 0; k < NUM_FEATURES; k++)
            {
                if (fvector[k] != 0.0)
                {
                    x[n].index = k + 1;
                    x[n].value = fvector[k];
                    n++;
                }
            }
            /*各画素を三つのクラスに分類したので，その各画素にラベルを割り当てている*/
            x[n].index = -1;
            label = (int)svm_predict(model, x);
            if (label == get_label(org, dec, i, j, num_class, th_list))
            {
                success++;
            }
            cls->val[i][j] = label;
            /*各分類の再生誤差の総計を求めている．あとでそのクラスに割り当てられた画素数でわり平均値をオフセット値とする*/
            offset[label] += org->val[i][j] - dec->val[i][j];
            cls_hist[label]++;
        }
        fprintf(stderr, ".");
    }

    fprintf(stderr, "\n");
    printf("Accuracy = %.2f (%%)\n", 100.0 * success / (dec->width * dec->height));
    side_info = 0;
    for (k = 0; k < num_class; k++)
    {
        if (cls_hist[k] > 0)
        {
            offset[k] /= cls_hist[k]; /*offset値は各クラスの再生誤差の平均値*/
        }
        printf("Offset[%d] = %.2f (%d)\n", k, offset[k], cls_hist[k]);
        offset[k] = n = floor(offset[k] + 0.5);
        if (n < 0)
            n = -n;
        side_info += (n + 1); // unary code
        if (n > 0)
            side_info++; // sign bit
    }
    /*各画素にオフセット値を加算する************************************************/
    for (i = 0; i < dec->height; i++)
    {
        for (j = 0; j < dec->width; j++)
        {
            label = cls->val[i][j];
            k = dec->val[i][j] + offset[label]; /*オフセット値の加算*/
            if (k < 0)
                k = 0;
            if (k > 255)
                k = 255;
            dec->val[i][j] = k;
            /**************************************************************************/
        }
    }
    printf("PSNR = %.3f (dB)\n", sn_after = calc_snr(org, dec)); /*pfsvmによる輝度補償の時のPSNR*/
    printf("GAIN = %+.3f (dB)\n", sn_after - sn_before);
    printf("SIDE_INFO = %d (bits)\n", side_info);
    write_pgm(dec, modimg);
    svm_free_and_destroy_model(&model);
    free(x);
    printf("cpu time: %.2f sec.\n", cpu_time());
    return (0);
}
