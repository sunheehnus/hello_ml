#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define TRAIN_MAX_SIZE 65536
#define THETA_MAX_SIZE 2048
#define THETA_MAX_CNT 32

#define NN_MAX_LEVEL 32
#define ACT_MAX_SIZE THETA_MAX_SIZE

#define TRAIN_SIZE 60000
#define THETA_SIZE (784 + 1)

double acts[NN_MAX_LEVEL][ACT_MAX_SIZE];
double thetas[NN_MAX_LEVEL][ACT_MAX_SIZE][ACT_MAX_SIZE];
double deltas[NN_MAX_LEVEL][ACT_MAX_SIZE];
double Deltas[NN_MAX_LEVEL][ACT_MAX_SIZE][ACT_MAX_SIZE];

double x[TRAIN_MAX_SIZE][THETA_MAX_SIZE];
double y[TRAIN_MAX_SIZE];

int layer_act_cnt[NN_MAX_LEVEL];

int debug = 1;

void init_thetas(int nn_level);
void back_propagation(int train_size, int nn_level, double alpha, double lambda);

static int load_train_data(char *train_images_path, char *train_labels_path);

void forward_propagation_step(int pos, int nn_level);
int main() {
    int i;
    load_train_data("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
    layer_act_cnt[0] = 784;
    layer_act_cnt[1] = 100;
    layer_act_cnt[2] = 10;
    init_thetas(3);
    for (i = 0; i < 10; i++)
        back_propagation(1000, 3, 0.01, 0.01);
    forward_propagation_step(100, 3);
    for (i = 0; i < 10; i++) {
        printf("%lf ", acts[2][i]);
    }
    printf("\n");
    return 0;
}
double vector_mul(double *vec1, double *vec2, int vec_size) {
    int i;
    double acc = 0;
    for (i = 0; i < vec_size; i++) {
        acc += vec1[i] * vec2[i];
    }
    return acc;
}
double sigmoid(double x) {
    return 1.0 / (1 + exp(-x));
}
double j_or_theta(int train_size, int nn_level) {
    int i;
    double res = 0;
    for (i = 0; i < train_size; i++) {
         for (j = 0; j < layer_act_cnt[nn_level - 1]; j++) {
             res += acts[nn_level - 1][j];
         }
    }
}
void init_thetas(int nn_level) {
    int i, j, k;
    for (i = 0; i < nn_level - 1; i++) {
        for (j = 0; j < layer_act_cnt[i + 1]; j++) {
            for (k = 0; k < layer_act_cnt[i]; k++) {
                thetas[i][j][k] = 2 * (double)rand() / (double)RAND_MAX - 1.0;
            }
        }
    }
}
void clear_deltas(int nn_level) {
    int i, j, k;
    for (i = 0; i < nn_level - 1; i++) {
        for (j = 0; j < layer_act_cnt[i + 1]; j++) {
            for (k = 0; k < layer_act_cnt[i]; k++) {
                Deltas[i][j][k] = 0;
            }
        }
    }
}
void forward_propagation_step(int pos, int nn_level) {
    int i, j;
    for (i = 0; i < layer_act_cnt[0]; i++) {
        acts[0][i] = x[pos][i];
    }
    for (i = 1; i < nn_level; i++) {
        for (j = 0; j < layer_act_cnt[i]; j++) {
            acts[i][j] = sigmoid(vector_mul(acts[i - 1], thetas[i - 1][j], layer_act_cnt[i - 1]));
        }
    }
}
void back_propagation_step(int pos, int nn_level) {
    int i, j, k;
    double theta_col[ACT_MAX_SIZE];
    for (i = 0; i < layer_act_cnt[nn_level - 1]; i++)
        deltas[nn_level - 1][i] = acts[nn_level - 1][i] - y[pos];
    for (i = nn_level - 2; i > 0; i--) {
        for (j = 0; j < layer_act_cnt[i]; j++) {
            for (k = 0; k < layer_act_cnt[i+1]; k++) {
                theta_col[k] = thetas[i][k][j];
            }
            deltas[i][j] = vector_mul(theta_col, deltas[i+1], layer_act_cnt[i + 1]) * acts[i][j] * (1 - acts[i][j]);
        }
    }
    for (i = 0; i < nn_level - 1; i++) {
        for (j = 0; j < layer_act_cnt[i + 1]; j++) {
            for (k = 0; k < layer_act_cnt[i]; k++) {
                Deltas[i][j][k] += acts[i][k] * deltas[i+1][j];
            }
        }
    }
}
void back_propagation(int train_size, int nn_level, double alpha, double lambda) {
    int i, j, k;
    clear_deltas(nn_level);
    for (i = 0; i < train_size; i++) {
        forward_propagation_step(i, nn_level);
        back_propagation_step(i, nn_level);
    }
    for (i = 0; i < nn_level - 1; i++) {
        for (j = 0; j < layer_act_cnt[i + 1]; j++) {
            for (k = 0; k < layer_act_cnt[i]; k++) {
                Deltas[i][j][k] /= train_size;
                if (k != 0)
                    Deltas[i][j][k] += lambda * thetas[i][j][k];
                thetas[i][j][k] -= Deltas[i][j][k] * alpha;
            }
        }
    }
}

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

    return TRAIN_SIZE;
}
