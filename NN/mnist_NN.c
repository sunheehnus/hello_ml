#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>

#define NN_MAX_LEVEL 32
#define TARIN_MAX_SIZE 65536
#define THETA_MAX_SIZE 2048
#define THETA_MAX_CNT 32

typedef struct Matrix {
    double *ptr;
    int M; // row cnt
    int N; // col cnt
} Matrix;

Matrix *activities[NN_MAX_LEVEL];
Matrix *t_activities[NN_MAX_LEVEL];

Matrix *theta[NN_MAX_LEVEL];
Matrix *t_theta[NN_MAX_LEVEL];

Matrix *bias[NN_MAX_LEVEL];
Matrix *t_bias[NN_MAX_LEVEL];

Matrix *deltas[NN_MAX_LEVEL];
Matrix *t_deltas[NN_MAX_LEVEL];

Matrix *deltas_theta[NN_MAX_LEVEL];
Matrix *deltas_bias[NN_MAX_LEVEL];

Matrix *delta_theta[NN_MAX_LEVEL];
Matrix *delta_bias[NN_MAX_LEVEL];

Matrix *adagrad_theta[NN_MAX_LEVEL];
Matrix *adagrad_bias[NN_MAX_LEVEL];

int layer_act_cnt[NN_MAX_LEVEL];
double x[TARIN_MAX_SIZE][THETA_MAX_SIZE];
double y[TARIN_MAX_SIZE][THETA_MAX_CNT];
int samples_pos[TARIN_MAX_SIZE];

static void load_train_data(char *train_images_path, char *train_labels_path, int train_size);
static void load_test_data(char *test_images_path, char *test_labels_path, int test_size);

void train_nn_bp(int iter_cnt, int train_size, int sample_size, int nn_layer, double alpha, double lambda);
double compute_accuracy(int cnt, int nn_layer);

int predict(int nn_layer, double x[]);

void test() {
    int nn_layer = 3;
    int iter_cnt = 1;
    int train_size = 1;
    int sample_size = 1;
    double alpha = 1;
    double lambda = 0;
    layer_act_cnt[0] = 2;
    layer_act_cnt[1] = 3;
    layer_act_cnt[2] = 1;
    x[0][0] = 0.8;
    x[0][1] = 0.2;
    y[0][0] = 1;
    train_nn_bp(iter_cnt, train_size, sample_size, nn_layer, alpha, lambda);
}
int main() {
    int nn_layer = 3;
    int iter_cnt = 1000;
    int train_size = 60000;
    int test_size = 500;
    int sample_size = 50;
    double alpha = 1;
    double lambda = 0;
    layer_act_cnt[0] = 784;
    layer_act_cnt[1] = 32;
    layer_act_cnt[2] = 10;

    load_train_data("train-images-idx3-ubyte", "train-labels-idx1-ubyte", train_size);
    train_nn_bp(iter_cnt, train_size, sample_size, nn_layer, alpha, lambda);

    load_test_data("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", test_size);

    printf("%lf\n", compute_accuracy(test_size, nn_layer));
    return 0;
}

#define PIX_CNT 784
static void load_train_data(char *train_images_path, char *train_labels_path, int train_size) {
    unsigned char buf[PIX_CNT];
    FILE *fp_ti = fopen(train_images_path, "r");
    FILE *fp_tl = fopen(train_labels_path, "r");
    int i, j;
    fseek(fp_ti, 16, SEEK_SET);
    fseek(fp_tl, 8, SEEK_SET);
    for (i = 0; i < train_size; i++) {
        fread(buf, PIX_CNT, 1, fp_ti);
        for (j = 0; j < PIX_CNT; j++) {
            /* if (j % 28 == 0) */
                /* printf("\n"); */
            x[i][j] = (double)buf[j]/255.0;
            /* printf("%.2lf", x[i][j]); */
        }
        fread(buf, 1, 1, fp_tl);
        y[i][buf[0]] = 1;
        /* printf("\ny: %d \n", buf[0]); */
    }
    fclose(fp_ti);
    fclose(fp_tl);
}
static void load_test_data(char *test_images_path, char *test_labels_path, int test_size) {
    int i;
    for (i = 0; i < test_size; i++) {
        memset(x[i], 0, sizeof(double) * PIX_CNT);
        memset(y[i], 0, sizeof(double) * 10);
    }
    load_train_data(test_images_path, test_labels_path, test_size);
    return;
}

Matrix * gen_matrix(int m, int n) {
    Matrix * out = (Matrix *)malloc(sizeof(Matrix));
    out->M = m;
    out->N = n;
    out->ptr = (double *)malloc(sizeof(double) * out->M * out->N);
    return out;
}
void destroy(Matrix *matrix) {
    free(matrix->ptr);
    free(matrix);
}
double get_matrix_m_n(Matrix *matrix, int m, int n) {
    return matrix->ptr[matrix->N * m + n];
}
void set_matrix_m_n(Matrix *matrix, int m, int n, double value) {
    matrix->ptr[matrix->N * m + n] = value;
}
Matrix *t_of_matrix(Matrix *in, Matrix *target) {
    int i, j;
    if (target && (in->M != target->N || in->N != target->M)) {
        perror("transpose target matrix mismatch!\n");
    }
    target = target == NULL ? gen_matrix(in->N, in->M) : target;
    for (i = 0; i < in->M; i++) {
        for (j = 0; j < in->N; j++) {
            set_matrix_m_n(target, j, i, get_matrix_m_n(in, i, j));
        }
    }
    return target;
}
Matrix *add_matrix(Matrix *m1, Matrix *m2, Matrix *target) {
    int i, j;
    if (m1->M != m2->M || m1->N != m2->N) {
        perror("matrix add mismatch!\n");
    }
    else if (target && (target->M != m1->M || target->N != m1->N)) {
        perror("add target matrix mismatch!\n");
    }
    target = target == NULL ? gen_matrix(m1->M, m1->N) : target;
    for (i = 0; i < target->M; i++) {
        for (j = 0; j < target->N; j++) {
            set_matrix_m_n(target, i, j, get_matrix_m_n(m1, i, j) + get_matrix_m_n(m2, i, j));
        }
    }
    return target;
}
Matrix *mul_matrix(Matrix *m1, Matrix *m2, Matrix *target) {
    int i, j, k;
    double acc;
    if (m1->N != m2->M) {
        perror("matrix mul mismatch!\n");
    }
    else if (target && (target->M != m1->M || target->N != m2->N)) {
        perror("mul target matrix mismatch!\n");
    }
    target = target == NULL ? gen_matrix(m1->M, m2->N) : target;
    for (i = 0; i < target->M; i++) {
        for (j = 0; j < target->N; j++) {
            for (acc = 0, k = 0; k < m1->N; k++) {
                acc += get_matrix_m_n(m1, i, k) * get_matrix_m_n(m2, k, j);
            }
            set_matrix_m_n(target, i, j, acc);
        }
    }
    return target;
}
Matrix *copy_matrix(Matrix *from, Matrix *to) {
    int i, j;
    if (to && (from->M != to->M || from->N != to->N)) {
        perror("matrix copy mismatch!\n");
    }
    to = to == NULL ? gen_matrix(from->M, from->N) : to;
    for (i = 0; i < from->M; i++) {
        for (j = 0; j < from->N; j++) {
            set_matrix_m_n(to, i, j, get_matrix_m_n(from, i, j));
        }
    }
    return to;
}
void fill_random_number_to_matrix(Matrix *m, double floor, double ceil) {
    int i;
    for (i = 0; i < m->M * m->N; i++) {
        m->ptr[i] = (double)rand() / (double)RAND_MAX * (ceil - floor) + floor;
    }
}
void fill_x_to_matrix(Matrix *m, double x) {
    int i;
    for (i = 0; i < m->M * m->N; i++) {
        m->ptr[i] = x;
    }
}

/* For Debug Usage */
void fill_ordered_number_to_matrix(Matrix *m) {
    int i;
    double x = 0.1;
    for (i = 0; i < m->M * m->N; i++, x += 0.1) {
        m->ptr[i] = x;
    }
}
void display_matrix(Matrix *m) {
    int i, j;
    for (i = 0; i < m->M; i++) {
        for (j = 0; j < m->N; j++) {
            printf("%lf ", get_matrix_m_n(m, i, j));
        }
        printf("\n");
    }
}

/* This is the function for random pick k elements from range(0, len)
 * Warning: before using it, you must do the initialization
 * initialization:
 *    for (i = 0; i < train_size; i++) {
 *        samples_pos[i] = i;
 *    }
 */
void pick(int k, int len) {
    int i, j, tmp;
    if (k > len)
        perror("sample cnt is too big");
    for (i = 0; i < k; i++) {
        j = rand() % (len - i);
        tmp = samples_pos[i + j];
        samples_pos[i + j] = samples_pos[i];
        samples_pos[i] = tmp;
    }
}

void init(int nn_layer, int train_size) {
    int i;
    double floor = 0;
    double ceil = 0.1;
    for (i = 0; i < nn_layer; i++) {
        activities[i] = gen_matrix(layer_act_cnt[i], 1);
        t_activities[i] = gen_matrix(1, layer_act_cnt[i]);

        deltas[i] = gen_matrix(layer_act_cnt[i], 1);
        t_deltas[i] = gen_matrix(1, layer_act_cnt[i]);
    }
    for (i = 0; i < nn_layer - 1; i++) {
        theta[i] = gen_matrix(layer_act_cnt[i + 1], layer_act_cnt[i]);
        fill_random_number_to_matrix(theta[i], floor, ceil);

        t_theta[i] = gen_matrix(layer_act_cnt[i], layer_act_cnt[i + 1]);

        bias[i] = gen_matrix(layer_act_cnt[i + 1], 1);
        fill_random_number_to_matrix(bias[i], floor, ceil);

        t_bias[i] = gen_matrix(1, layer_act_cnt[i + 1]);

        delta_theta[i] = gen_matrix(layer_act_cnt[i + 1], layer_act_cnt[i]);
        delta_bias[i] = gen_matrix(layer_act_cnt[i + 1], 1);

        deltas_theta[i] = gen_matrix(layer_act_cnt[i + 1], layer_act_cnt[i]);
        deltas_bias[i] = gen_matrix(layer_act_cnt[i + 1], 1);

        adagrad_theta[i] = gen_matrix(layer_act_cnt[i + 1], layer_act_cnt[i]);
        fill_x_to_matrix(adagrad_theta[i], 0.000001);
        adagrad_bias[i] = gen_matrix(layer_act_cnt[i + 1], 1);
        fill_x_to_matrix(adagrad_bias[i], 0.000001);
    }
    /* init the samples_pos for selecting random gradient samples */
    for (i = 0; i < train_size; i++) {
        samples_pos[i] = i;
    }
}
void clear_deltas(int nn_layer) {
    int i, j, k;
    for (i = 0; i < nn_layer - 1; i++) {
        for (j = 0; j < deltas_theta[i]->M; j++) {
            for (k = 0; k < deltas_theta[i]->N; k++) {
                set_matrix_m_n(deltas_theta[i], j, k, 0);
            }
        }
        for (j = 0; j < deltas_bias[i]->M; j++) {
            for (k = 0; k < deltas_bias[i]->N; k++) {
                set_matrix_m_n(deltas_bias[i], j, k, 0);
            }
        }
    }
}
static double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
void forward_propagation_step(int pos, int nn_layer) {
    int i, j;
    for (i = 0; i < activities[0]->M; i++) {
        set_matrix_m_n(activities[0], i, 0, x[pos][i]);
    }
    for (i = 1; i < nn_layer; i++) {
        mul_matrix(theta[i - 1], activities[i - 1], activities[i]);
        add_matrix(activities[i], bias[i - 1], activities[i]);
        /* sigmoid the inputs */
        for (j = 0; j < activities[i]->M; j++) {
            set_matrix_m_n(activities[i], j, 0, sigmoid(get_matrix_m_n(activities[i], j, 0)));
        }
    }
}
void backward_propagation_step(int pos, int nn_layer) {
    int i, j;
    double dfz = 0;
    /* initialize the last deltas */
    for (i = 0; i < deltas[nn_layer - 1]->M; i++)
        set_matrix_m_n(deltas[nn_layer - 1], i, 0, get_matrix_m_n(activities[nn_layer - 1], i, 0) - y[pos][i]);
    /* compute all the left deltas */
    for (i = nn_layer - 2; i > 0; i--) {
        t_of_matrix(theta[i], t_theta[i]);
        mul_matrix(t_theta[i], deltas[i + 1], deltas[i]);
        /* dot multi */
        for (j = 0; j < deltas[i]->M; j++) {
            dfz = get_matrix_m_n(activities[i], j, 0) * (1 - get_matrix_m_n(activities[i], j, 0));
            set_matrix_m_n(deltas[i], j, 0, get_matrix_m_n(deltas[i], j, 0) * dfz);
        }
    }
    for (i = 0; i < nn_layer - 1; i++) {
        t_of_matrix(activities[i], t_activities[i]);
        mul_matrix(deltas[i + 1], t_activities[i], delta_theta[i]);
        copy_matrix(deltas[i+1], delta_bias[i]);
    }
}
void update(int sample_size, int nn_layer, double alpha, int lambda) {
    int l, i, j;
    double former, d;
    for (l = 0; l < nn_layer - 1; l++) {
        for (i = 0; i < theta[l]->M; i++) {
            for (j = 0; j < theta[l]->N; j++) {
                former = get_matrix_m_n(theta[l], i, j);

                d = get_matrix_m_n(deltas_theta[l], i, j) / sample_size + lambda * former;
                /* set_matrix_m_n(adagrad_theta[l], i, j, get_matrix_m_n(adagrad_theta[l], i, j) + d * d); */

                /* set_matrix_m_n(theta[l], i, j, former - alpha * d / sqrt(get_matrix_m_n(adagrad_theta[l], i, j))); */
                set_matrix_m_n(theta[l], i, j, former - alpha * d);
            }
        }
        for (i = 0; i < bias[l]->M; i++) {
            for (j = 0; j < bias[l]->N; j++) {
                former = get_matrix_m_n(bias[l], i, j);

                d = get_matrix_m_n(deltas_bias[l], i, j) / sample_size + lambda * former;
                /* set_matrix_m_n(adagrad_bias[l], i, j, get_matrix_m_n(adagrad_bias[l], i, j) + d * d); */

                /* set_matrix_m_n(bias[l], i, j, former - alpha * d / sqrt(get_matrix_m_n(adagrad_bias[l], i, j))); */
                set_matrix_m_n(bias[l], i, j, former - alpha * d);
            }
        }
    }
}
double j_of_theta(int train_size, int nn_layer) {
    int i, j;
    double res = 0;
    for (i = 0; i < train_size; i++) {
        forward_propagation_step(i, nn_layer);
        for (j = 0; j < activities[nn_layer - 1]->M; j++) {
            res -= (1 - y[i][j]) * log(1 - get_matrix_m_n(activities[nn_layer - 1], j, 0)) +
                y[i][j] * log(get_matrix_m_n(activities[nn_layer - 1], j, 0));
        }
    }
    return res / train_size;
}
double j_of_theta_sample(int sample_size, int nn_layer) {
    int i, j;
    double res = 0;
    for (i = 0; i < sample_size; i++) {
        forward_propagation_step(samples_pos[i], nn_layer);
        for (j = 0; j < activities[nn_layer - 1]->M; j++) {
            res -= (1 - y[samples_pos[i]][j]) * log(1 - get_matrix_m_n(activities[nn_layer - 1], j, 0)) +
                y[samples_pos[i]][j] * log(get_matrix_m_n(activities[nn_layer - 1], j, 0));
        }
    }
    return res / sample_size;
}
void gradient_checking(int nn_layer, int sample_size) {
    int i, j, k;
    double res = 0;
    for (i = nn_layer - 2; i < nn_layer - 1; i++) {
        for (j = 0; j < theta[i]->M; j++) {
            for (k = 0; k < theta[i]->N; k++) {
                set_matrix_m_n(theta[i], j, k, get_matrix_m_n(theta[i], j, k) + 0.0001);
                res = j_of_theta_sample(sample_size, nn_layer);
                set_matrix_m_n(theta[i], j, k, get_matrix_m_n(theta[i], j, k) - 0.0002);
                res -= j_of_theta_sample(sample_size, nn_layer);
                set_matrix_m_n(theta[i], j, k, get_matrix_m_n(theta[i], j, k) + 0.0001);
                printf("%lf %lf\n", res / 0.0002, get_matrix_m_n(deltas_theta[i], j, k) / sample_size);
            }
        }
    }
}
void backward_propagation(int sample_size, int nn_layer, double alpha, double lambda) {
    int i, j;
    clear_deltas(nn_layer);
    for (i = 0; i < sample_size; i++) {
        forward_propagation_step(samples_pos[i], nn_layer);
        backward_propagation_step(samples_pos[i], nn_layer);
        for (j = 0; j < nn_layer - 1; j++) {
            add_matrix(deltas_theta[j], delta_theta[j], deltas_theta[j]);
            add_matrix(deltas_bias[j], delta_bias[j], deltas_bias[j]);
        }
    }
    gradient_checking(nn_layer, sample_size);
    update(sample_size, nn_layer, alpha, lambda);
}
void train_nn_bp(int iter_cnt, int train_size, int sample_size, int nn_layer, double alpha, double lambda) {
    init(nn_layer, train_size);
    while (iter_cnt--) {
        pick(sample_size, train_size);
        backward_propagation(sample_size, nn_layer, alpha, lambda);
        /* printf("j of theta: %lf\n", j_of_theta(train_size, nn_layer)); */
    }
}

int predict(int nn_layer, double x[]) {
    int i, j;
    int max_pos = -1;
    double max = -1008611;
    printf("%d\n", activities[0]->M);
    for (i = 0; i < activities[0]->M; i++) {
        set_matrix_m_n(activities[0], i, 0, x[i]);
    }
    for (i = 1; i < nn_layer; i++) {
        mul_matrix(theta[i - 1], activities[i - 1], activities[i]);
        add_matrix(activities[i], bias[i - 1], activities[i]);
        /* sigmoid the inputs */
        for (j = 0; j < activities[i]->M; j++) {
            set_matrix_m_n(activities[i], j, 0, sigmoid(get_matrix_m_n(activities[i], j, 0)));
        }
    }
    for (i = 0; i < activities[nn_layer - 1]->M; i++) {
        if (get_matrix_m_n(activities[nn_layer - 1], i, 0) > max) {
            max_pos = i;
            max = get_matrix_m_n(activities[nn_layer - 1], i, 0);
        }
    }
    return max_pos;
}

int answer(int pos, int nn_layer) {
    int i;
    int max_pos = -1;
    double max = -1008611;
    forward_propagation_step(pos, nn_layer);
    for (i = 0; i < activities[nn_layer - 1]->M; i++) {
        if (get_matrix_m_n(activities[nn_layer - 1], i, 0) > max) {
            max_pos = i;
            max = get_matrix_m_n(activities[nn_layer - 1], i, 0);
        }
    }
    return max_pos;
}
int get_y(int pos) {
    int i;
    for (i = 0; i <= 9; i++) {
        if (1.1 - y[pos][i] < 0.2)
            return i;
    }
    return -1;
}

double compute_accuracy(int cnt, int nn_layer) {
    int i;
    int right = 0;
    int ans, y;
    for (i = 0; i < cnt; i++)
    {
        ans = answer(i, nn_layer);
        y = get_y(i);
        right += ans == y;
    }
    return 1.0 * right / cnt;
}
