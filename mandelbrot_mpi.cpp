#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Tags das mensagens MPI
#define TAG_REQUEST 0
#define TAG_WORK    1
#define TAG_STOP    2
#define TAG_RESULT  3
#define TAG_STATS   4

// Mesmo cálculo de Mandelbrot usado no T1
static int mandelbrot(double cx, double cy, int max_iter) {
    double zx = 0.0, zy = 0.0;
    int iter = 0;

    while (zx * zx + zy * zy <= 4.0 && iter < max_iter) {
        double next_zx = zx * zx - zy * zy + cx;
        zy = 2.0 * zx * zy + cy;
        zx = next_zx;
        ++iter;
    }

    return iter;
}

// Confere os argumentos passados pelo terminal
static int parse_positive(const char *s, int *out) {
    char *end = nullptr;
    long v = strtol(s, &end, 10);

    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT_MAX)
        return 0;

    *out = (int)v;
    return 1;
}

// Gera o PNG somente se solicitado (fora da medição)
static void save_png(const int *values, int width, int height,
                     int max_iter, const char *path) {
    size_t pixels = (size_t)width * height;
    std::vector<uint8_t> image(pixels * 3);

    for (size_t i = 0; i < pixels; ++i) {
        int iter = values[i];
        double t = (double)iter / max_iter;

        if (iter == max_iter) {
            image[i * 3] = image[i * 3 + 1] = image[i * 3 + 2] = 0;
        } else {
            image[i * 3]     = (uint8_t)(9 * (1 - t) * t * t * t * 255);
            image[i * 3 + 1] = (uint8_t)(15 * (1 - t) * (1 - t) * t * t * 255);
            image[i * 3 + 2] = (uint8_t)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
        }
    }

    if (!stbi_write_png(path, width, height, 3, image.data(), width * 3))
        std::fprintf(stderr, "Falha ao salvar PNG: %s\n", path);

}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Valores padrao, podem ser alterados via terminal
    int width = 2000, height = 2000, max_iter = 1000, chunk = 1;
    int png = 0, valid = nprocs >= 2;

    if (argc != 1 && argc != 5 && argc != 6)
        valid = 0;

    if (argc >= 5) {
        valid &= parse_positive(argv[1], &width);
        valid &= parse_positive(argv[2], &height);
        valid &= parse_positive(argv[3], &max_iter);
        valid &= parse_positive(argv[4], &chunk);
    }

    if (argc == 6 && strcmp(argv[5], "--png") != 0)
        valid = 0;
    if (argc == 6)
        png = 1;

    // Evita tamanhos invalidos ou buffers acima do limite do MPI
    if (width < 2 || height < 2 ||
        (size_t)width * height > SIZE_MAX / sizeof(int) ||
        (size_t)width * (chunk < height ? chunk : height) > (size_t)INT_MAX - 2 ||
        (size_t)width * height > SIZE_MAX / 3)
        valid = 0;

    if (!valid) {
        if (rank == 0)
            std::fprintf(stderr, "Uso: mpirun -np N ./mandelbrot_mpi [largura altura maxIter chunk [--png]] (N >= 2)\n");

        MPI_Finalize();
        return 1;
    }

    if (chunk > height)
        chunk = height;

    const size_t pixels = (size_t)width * height;

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    if (rank == 0) {
        // Coordenador: guarda a imagem e controla as proximas linhas
        std::vector<int> values(pixels);
        std::vector<uint8_t> seen(height, 0);
        std::vector<int> result((size_t)width * chunk + 2);
        std::vector<double> all_stats((size_t)nprocs * 3, 0.0);

        int next_row = 0, rows_done = 0, stopped = 0, stats_received = 0;

        // Atende os pedidos e recebe os resultados conforme chegam
        while (stats_received < nprocs - 1) {
            MPI_Status status;
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            int source = status.MPI_SOURCE;

            if (status.MPI_TAG == TAG_REQUEST) {
                int request;
                MPI_Recv(&request, 1, MPI_INT, source, TAG_REQUEST,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (next_row < height) {
                    // Envia um bloco de linhas para quem pediu trabalho
                    int work[2] = {
                        next_row,
                        (height - next_row < chunk ? height - next_row : chunk)
                    };

                    next_row += work[1];
                    MPI_Send(work, 2, MPI_INT, source, TAG_WORK, MPI_COMM_WORLD);
                } else {
                    // Acabaram as tarefas: avisa esse trabalhador para parar
                    int stop = -1;
                    MPI_Send(&stop, 1, MPI_INT, source, TAG_STOP, MPI_COMM_WORLD);
                    ++stopped;
                }

            } else if (status.MPI_TAG == TAG_RESULT) {
                int count;
                MPI_Get_count(&status, MPI_INT, &count);

                if (count < 3 || count > 2 + width * chunk)
                    MPI_Abort(MPI_COMM_WORLD, 3);

                MPI_Recv(result.data(), count, MPI_INT, source, TAG_RESULT,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                int first = result[0], rows = result[1];

                if (first < 0 || rows < 1 || first > height - rows ||
                    count != rows * width + 2)
                    MPI_Abort(MPI_COMM_WORLD, 4);

                // Marca cada linha recebida, evitando duplicatas
                for (int y = first; y < first + rows; ++y) {
                    if (seen[y])
                        MPI_Abort(MPI_COMM_WORLD, 5);
                    seen[y] = 1;
                }

                // Copia o bloco para a posicao original na imagem
                memcpy(values.data() + (size_t)first * width, result.data() + 2,
                       (size_t)rows * width * sizeof(int));
                rows_done += rows;

            } else if (status.MPI_TAG == TAG_STATS) {
                // Estatisticas enviadas quando o trabalhador termina
                MPI_Recv(all_stats.data() + 3 * source, 3, MPI_DOUBLE, source,
                         TAG_STATS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                ++stats_received;

            } else {
                MPI_Abort(MPI_COMM_WORLD, 6);
            }
        }

        double elapsed = MPI_Wtime() - start;

        if (rows_done != height || stopped != nprocs - 1)
            MPI_Abort(MPI_COMM_WORLD, 7);

        long long checksum = 0;
        for (size_t i = 0; i < pixels; ++i)
            checksum += values[i];

        std::printf("Tempo MPI: %.9f segundos (%d processos, %d trabalhadores)\n",
               elapsed, nprocs, nprocs - 1);
        std::printf("Configuracao: %d x %d, maxIter=%d, chunk=%d\n",
               width, height, max_iter, chunk);
        std::printf("Checksum: %lld\n", checksum);
        std::printf("rank,linhas,chunks,tempo_calculo_s\n");

        for (int p = 1; p < nprocs; ++p)
            std::printf("%d,%.0f,%.0f,%.9f\n", p, all_stats[p * 3],
                   all_stats[p * 3 + 1], all_stats[p * 3 + 2]);

        if (png)
            save_png(values.data(), width, height, max_iter, "mandelbrot_mpi.png");


    } else {
        // Trabalhadores: pedem, calculam e devolvem chunks ao rank 0
        std::vector<int> buffer((size_t)width * chunk + 2);

        // linhas, chunks e tempo gasto so nos calculos
        double stats[3] = {0.0, 0.0, 0.0};

        while (1) {
            int request = 1, work[2] = {-1, 0};
            MPI_Status status;

            MPI_Send(&request, 1, MPI_INT, 0, TAG_REQUEST, MPI_COMM_WORLD);
            MPI_Recv(work, 2, MPI_INT, 0, MPI_ANY_TAG,
                     MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_STOP)
                break;

            if (status.MPI_TAG != TAG_WORK || work[0] < 0 ||
                work[1] < 1 || work[1] > chunk)
                MPI_Abort(MPI_COMM_WORLD, 9);

            // Cabecalho do resultado: primeira linha e qtd de linhas
            buffer[0] = work[0];
            buffer[1] = work[1];
            double calc_start = MPI_Wtime();

            for (int j = 0; j < work[1]; ++j) {
                int y = work[0] + j;
                double minY = -1.5, maxY = 1.5;
                double cy = minY + (maxY - minY) * y / (height - 1.0);

                for (int x = 0; x < width; ++x) {
                    double minX = -2.0, maxX = 1.0;
                    double cx = minX + (maxX - minX) * x / (width - 1.0);

                    buffer[2 + (size_t)j * width + x] = mandelbrot(cx, cy, max_iter);
                }
            }

            stats[2] += MPI_Wtime() - calc_start;
            stats[0] += work[1];
            stats[1] += 1;

            MPI_Send(buffer.data(), 2 + work[1] * width, MPI_INT, 0,
                     TAG_RESULT, MPI_COMM_WORLD);
        }

        MPI_Send(stats, 3, MPI_DOUBLE, 0, TAG_STATS, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
