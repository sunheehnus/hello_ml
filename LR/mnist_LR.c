#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>

#define TRAIN_MAX_SIZE 65536
#define THETA_MAX_SIZE 2048
#define THETA_MAX_CNT 32

#define TRAIN_SIZE 60000
#define THETA_SIZE (784 + 1)
#define THETA_CNT 10


int debug = 1;

double x[TRAIN_MAX_SIZE][THETA_MAX_SIZE];
double y[TRAIN_MAX_SIZE];
double theta[THETA_MAX_CNT][THETA_MAX_SIZE];
double delta[THETA_MAX_CNT][THETA_MAX_SIZE];
double gradient[THETA_MAX_CNT][THETA_MAX_SIZE];
int train_data_size;
int samples_pos[TRAIN_MAX_SIZE];

#define EQUAL(x, y) (((x) - (y) > -0.0001) && ((x) - (y) < 0.0001))

#define PIX_CNT 784
static int load_train_data(char *train_images_path, char *train_labels_path) {
    unsigned char buf[PIX_CNT];
    FILE *fp_ti = fopen(train_images_path, "r");
    FILE *fp_tl = fopen(train_labels_path, "r");
    int i, j;

    fseek(fp_ti, 16, SEEK_SET);
    fseek(fp_tl, 8, SEEK_SET);
    for (i = 0; i < TRAIN_SIZE; i++) {
        fread(buf, PIX_CNT, 1, fp_ti);
        x[i][0] = 1;
        for (j = 0; j < PIX_CNT; j++) {
            x[i][j+1] = buf[j];
        }
        fread(buf, 1, 1, fp_tl);
        y[i] = buf[0];
    }
    fclose(fp_ti);
    fclose(fp_tl);

    train_data_size = TRAIN_SIZE;
    return TRAIN_SIZE;
}
static void save_theta_data(char *path, int theta_cnt, int theta_size) {
    FILE *fp = fopen(path, "w");
    int i, j;
    for (i = 0; i< theta_cnt; i++) {
        for (j = 0; j < theta_size; j++)
            fprintf(fp, "%lf\t", theta[i][j]);
        fprintf(fp, "\n");
    }
    fprintf(fp, "\n");
}

/* pick ceil elements */
void init_pick(int ceil) {
    int i;
    for (i = 0; i < ceil; i++) {
        samples_pos[i] = i;
    }
}
void pick(int samples_pos[], int k, int len) {
    int i, j, tmp;
    if (k > len)
        exit(1);
    for (i = 0; i < k; i++) {
        j = rand() % (len - i);
        tmp = samples_pos[i + j];
        samples_pos[i + j] = samples_pos[i];
        samples_pos[i] = tmp;
    }
}
/* End Of pick ceil elements */

double vector_mul(double vec1[], double vec2[], int length) {
    int i;
    double res = 0;
    for (i = 0; i < length; i++)
        res += vec1[i] * vec2[i];
    return res;
}

double probablity(int i, int j, int theta_size, int theta_cnt) {
    int l;
    double tmp = vector_mul(theta[j], x[i], theta_size);
    double div_down = 0;
    for (l = 0; l < theta_cnt; l++) {
        div_down += exp(vector_mul(theta[l], x[i], theta_size) - tmp);
    }
    /* printf("up: %lf, down: %lf\n", div_up, div_down); */
    return 1.0 / div_down;
}

double j_of_theta(train_data_size, theta_cnt, theta_size) {
    int i, j, k;
    for (i = 0; i < theta_cnt; i++) {
    }
    return 0;
}

double _abs(double a) {
    return a > 0 ? a : -a;
}
void train_softmax_stoc(int iter_cnt, double alpha, double lambda, int theta_cnt, int theta_size, int train_data_size, int sample_cnt) {
    int i, j, k;
    double scale;
    init_pick(train_data_size);
    while (iter_cnt--) {
        pick(samples_pos, sample_cnt, train_data_size);
        for (j = 0; j < theta_cnt; j++) {
            for (k = 0; k < theta_size; k++) {
                delta[j][k] = 0;
            }
            for (i = 0; i < sample_cnt; i++) {
                scale = -probablity(samples_pos[i], j, theta_size, theta_cnt) + EQUAL(y[samples_pos[i]], j);
                for (k = 0; k < theta_size; k++) {
                    delta[j][k] += x[samples_pos[i]][k] * scale;
                }
            }
            for (k = 0; k < theta_size; k++) {
                delta[j][k] /= -sample_cnt;
                delta[j][k] += theta[j][k] * lambda;
                gradient[j][k] += delta[j][k] * delta[j][k];
            }
        }
        for (j = 0; j < theta_cnt; j++) {
            for (k = 0; k < theta_size; k++) {
                if (_abs(gradient[j][k] > 0.000001))
                    theta[j][k] -= alpha * delta[j][k] / sqrt(gradient[j][k]);
            }
        }
        if (debug) {
            /* printf("%lf\n", j_of_theta(train_data_size, theta_size)); */
        }
    }
}

int main() {
    int i, j;
    double h_theta;
    /* debug = 0; */
    load_train_data("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
    printf("load successfully!\n");

    train_softmax_stoc(5000, 0.01, 0.01, THETA_CNT, THETA_SIZE, TRAIN_SIZE, 50);
    save_theta_data("mmnniisstt", THETA_CNT, THETA_SIZE);
    return 0;
}
