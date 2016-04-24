#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define TRAIN_MAX_SIZE 65536
#define THETA_MAX_SIZE 2048

#define MIN_POSITIVE_NUM 0.00000001

int debug = 1;

double x[TRAIN_MAX_SIZE][THETA_MAX_SIZE];
double y[TRAIN_MAX_SIZE];
double theta[THETA_MAX_SIZE];
double gradient[THETA_MAX_SIZE];
int train_data_size;
int samples_pos[TRAIN_MAX_SIZE];

/* pick ceil elements */
void init_pick(int ceil);
void pick(int samples_pos[], int k, int len);
/* End Of pick ceil elements */

/* load train data */
static char *next_pos(char *str);
static int load_train_data(char *path, int theta_cnt);
static void save_theta_data(char *path, int theta_cnt);
/* End Of load train data */

double sigmoid(double x) {
    return 1.0 / (1 + exp(-x));
}

double vector_mul(double vec1[], double vec2[], int length) {
    int i;
    double res = 0;
    for (i = 0; i < length; i++)
        res += vec1[i] * vec2[i];
    return res;
}

double j_of_theta(int train_data_size, int theta_cnt) {
    int i;
    double h_theta;
    double res = 0;
    for (i = 0; i < train_data_size; i++) {
        h_theta = sigmoid(vector_mul(theta, x[i], theta_cnt));
        res += y[i] * log10(h_theta) + (1 - y[i]) * log10(1 - h_theta);
    }
    return -res / train_data_size;
}

void train_LR_stoc(int iter_cnt, double alpha, int theta_cnt, int train_data_size, int sample_cnt) {
    int i, j;
    double deltas[THETA_MAX_SIZE];
    init_pick(train_data_size);
    while (iter_cnt--) {
        pick(samples_pos, sample_cnt, train_data_size);
        for (j = 0; j < theta_cnt; j++) {
            deltas[j] = 0;
            for (i = 0; i < sample_cnt; i++) {
                deltas[j] += (sigmoid(vector_mul(x[samples_pos[i]], theta, theta_cnt)) - y[samples_pos[i]]) * x[samples_pos[i]][j];
            }
            deltas[j] /= train_data_size;
            gradient[j] += deltas[j] * deltas[j];
        }
        for (j = 0; j < theta_cnt; j++) {
            if (gradient[j] > MIN_POSITIVE_NUM)
                theta[j] -= alpha * deltas[j] / sqrt(gradient[j]);
        }
        if (debug) {
            printf("%lf\n", j_of_theta(train_data_size, theta_cnt));
        }
    }
}

int main() {
    int i;
    double h_theta;
    /* debug = 0; */
    load_train_data("train_data", 3);
    train_LR_stoc(1000000, 1, 3, train_data_size, 10);
    save_theta_data("theta_data", 3);
    printf("%lf %lf %lf\n", theta[0], theta[1], theta[2]);
    for (i = 0; i < train_data_size; i++) {
        h_theta = sigmoid(vector_mul(theta, x[i], 3));
        printf("%lf %lf %lf %lf\n", x[i][1], x[i][2], y[i], h_theta);
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

/* load train data related */
static char *next_pos(char *str) {
    while (*str == ' ' || *str == '\t')
        str++;
    while (*str != '\0' && *str != ' ' && *str != '\t')
        str++;
    return str;
}
static int load_train_data(char *path, int theta_cnt) {
    FILE *fp = fopen(path, "r");
    char buf[2048], *cur;
    int pos = 0;
    int i;
    while(fgets(buf, 2048, fp) != NULL) {
        x[pos][0] = 1;
        cur = buf;
        for (i = 1; i < theta_cnt; i++) {
            sscanf(cur, "%lf", &x[pos][i]);
            cur = next_pos(cur);
        }
        sscanf(cur, "%lf", &y[pos]);
        pos++;
    }
    train_data_size = pos;
    return pos;
}
static void save_theta_data(char *path, int theta_cnt) {
    FILE *fp = fopen(path, "w");
    int i;
    for (i = 0; i < theta_cnt; i++)
        fprintf(fp, "%lf\t", theta[i]);
    fprintf(fp, "\n");
}
/* End Of load train data related */
