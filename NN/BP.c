#include<stdio.h>
#include<stdlib.h>
#include<string.h>
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
double y[TRAIN_MAX_SIZE][THETA_MAX_CNT];

int layer_act_cnt[NN_MAX_LEVEL];

int samples_pos[TRAIN_MAX_SIZE];

int debug = 1;

void init_thetas(int nn_level);
void back_propagation(int train_size, int nn_level, double alpha, double lambda);

static int load_train_data(char *train_images_path, char *train_labels_path);

double compute_accuracy(int cnt, int nn_level);

int main() {
    int i;
    int nn_level = 3;
    load_train_data("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
    layer_act_cnt[0] = 784;
    layer_act_cnt[1] = 128;
    layer_act_cnt[2] = 10;
    /* layer_act_cnt[2] = 128; */
    /* layer_act_cnt[3] = 10; */
    init_thetas(nn_level);
    for (i = 0; i < 10; i++)
        back_propagation(6000, nn_level, 0.1, 0.01);
    printf("%lf\n", compute_accuracy(100, nn_level));
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
double j_of_theta(int train_size, int nn_level, double lambda) {
    int i, j, l;
    double res = 0;
    for (i = 0; i < train_size; i++) {
         for (j = 0; j < layer_act_cnt[nn_level - 1]; j++) {
             res -= (1 - y[nn_level - 1][j]) * log10(1 - acts[nn_level - 1][j]) + y[nn_level - 1][j] * log10(acts[nn_level - 1][j]);
         }
    }
    for (l = 0; l < nn_level - 1; l++) {
        for (i = 0; i < layer_act_cnt[l]; i++) {
            for (j = 0; j < layer_act_cnt[l + 1]; j++)
                res += lambda * thetas[l][i][j] * thetas[l][i][j] / 2;
        }
    }
    return res / train_size;
}
void init_thetas(int nn_level) {
    int i, j, k;
    for (i = 0; i < nn_level - 1; i++) {
        for (j = 0; j < layer_act_cnt[i + 1]; j++) {
            for (k = 0; k < layer_act_cnt[i]; k++) {
                thetas[i][j][k] = 2.0 * (double)rand() / (double)RAND_MAX - 1.0;
            }
        }
    }
}
void clear_deltas(int nn_level) {
    int i, j, k;
    for (i = 0; i < nn_level - 1; i++) {
        for (j = 0; j < layer_act_cnt[i + 1]; j++) {
            /* memset(Deltas, 0, sizeof(double) * layer_act_cnt[i]); */
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
        deltas[nn_level - 1][i] = acts[nn_level - 1][i] - y[pos][i];
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
    printf("%lf\n", j_of_theta(train_size, nn_level, lambda));
}

void train_NN_BP(int iter_cnt, int train_size, int sample_cnt, int nn_level, double alpha, double lambda) {
    init_pick(train_size);
    while (iter_cnt --) {
        pick(samples_pos, sample_cnt, train_size);
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
        y[i][buf[0]] = 1;
    }
    fclose(fp_ti);
    fclose(fp_tl);

    return TRAIN_SIZE;
}

int isRight(int pos, int nn_level) {
    int i;
    int y_max_pos, guess_max_pos;
    double y_max, guess_max;
    y_max_pos = guess_max_pos = -1;
    y_max = guess_max = -10086;

    forward_propagation_step(pos, nn_level);
    for (i = 0; i < layer_act_cnt[nn_level - 1]; i++) {
        if (y[pos][i] > y_max_pos) {
            y_max_pos = i;
            y_max = y[pos][y_max_pos];
        }
        if (acts[nn_level - 1][i] > guess_max) {
            guess_max_pos = i;
            guess_max = acts[nn_level - 1][y_max_pos];
        }
    }
    return guess_max_pos == y_max_pos;
}
double compute_accuracy(int cnt, int nn_level) {
    int i;
    int right = 0;
    for (i = 0; i < cnt; i++)
        right += isRight(i, nn_level);
    return 1.0 * right / cnt;
}
