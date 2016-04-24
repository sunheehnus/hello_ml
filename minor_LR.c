#include<stdio.h>
#include<stdlib.h>
#include<math.h>

double x[65536][2048];
double y[65536];
double theta[2048];
int train_data_size;

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

double j_of_theta(int train_data_size) {
    int i;
    double h_theta;
    double res = 0;
    for (i = 0; i < train_data_size; i++) {
        h_theta = sigmoid(vector_mul(theta, x[i], 3));
        res += y[i] * log10(h_theta) + (1 - y[i]) * log10(1 - h_theta);
    }
    return -res / train_data_size;
}

void trainLogRegres(int iter_cnt, double alpha, int theta_cnt, int train_data_size) {
    int i, j;
    double deltas[2048];
    while (iter_cnt--) {
        for (i = 0; i < theta_cnt; i++) {
            deltas[i] = 0;
            for (j = 0; j < train_data_size; j++) {
                deltas[i] += (sigmoid(vector_mul(x[j], theta, theta_cnt)) - y[j]) * x[j][i];
            }
            deltas[i] /= train_data_size;
        }
        for (i = 0; i < theta_cnt; i++) {
            theta[i] -= alpha * deltas[i];
        }
    }
}

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

int main() {
    int i;
    double h_theta;
    load_train_data("train_data", 3);
    trainLogRegres(100000, 1, 3, train_data_size);
    save_theta_data("theta_data", 3);
    printf("%lf %lf %lf\n", theta[0], theta[1], theta[2]);
    for (i = 0; i < train_data_size; i++) {
        h_theta = sigmoid(vector_mul(theta, x[i], 3));
        printf("%lf %lf %lf %lf\n", x[i][1], x[i][2], y[i], h_theta);
    }
}
