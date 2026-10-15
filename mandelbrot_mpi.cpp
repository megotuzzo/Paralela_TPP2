#include <iostream>
#include <vector>
#include <cstdint>
#include <mpi.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define TAG_REQUEST 0
#define TAG_WORK 1
#define TAG_STOP 2
#define TAG_RESULT_BASE 10

int mandelbrot(double cx, double cy, int maxIter) {
    double zx = 0.0, zy = 0.0;
    int iter = 0;
    while (zx * zx + zy * zy <= 4.0 && iter < maxIter) {
        double novoZx = zx * zx - zy * zy + cx;
        zy = 2.0 * zx * zy + cy;
        zx = novoZx;
        ++iter;
    }
    return iter;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int my_rank, proc_n;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_n);

    if (proc_n < 2) {
        if (my_rank == 0) std::cerr << "Precisa de pelo menos 2 processos!\n";
        MPI_Finalize();
        return 1;
    }

    const int largura = 2000;
    const int altura = 2000;
    const int maxIter = 1000; // Mesmo limite da versao sequencial do T1
    const double minX = -2.0, maxX = 1.0;
    const double minY = -1.5, maxY = 1.5;

    // TAG_RESULT_BASE + y deve estar dentro do limite de tags do MPI.
    int *tag_ub = nullptr, existe_limite = 0;
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_ub, &existe_limite);
    if (!existe_limite || TAG_RESULT_BASE + altura - 1 > *tag_ub) {
        if (my_rank == 0) std::cerr << "Intervalo de tags MPI insuficiente para a altura escolhida.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double inicio = MPI_Wtime();
    MPI_Status status;

    if (my_rank == 0) {
        std::vector<int> iteracoes(static_cast<size_t>(largura) * altura);
        std::vector<int> linhas_por_worker(proc_n, 0);
        int proxima_linha = 0;
        int linhas_concluidas = 0;
        int trabalhadores_finalizados = 0;

        // Atende solicitacoes e resultados na ordem de chegada.
        while (trabalhadores_finalizados < proc_n - 1) {
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            const int worker = status.MPI_SOURCE;
            const int tag = status.MPI_TAG;

            if (tag == TAG_REQUEST) {
                int pedido;
                MPI_Recv(&pedido, 1, MPI_INT, worker, TAG_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (proxima_linha < altura) {
                    const int y = proxima_linha++;
                    MPI_Send(&y, 1, MPI_INT, worker, TAG_WORK, MPI_COMM_WORLD);
                } else {
                    const int fim = -1;
                    MPI_Send(&fim, 1, MPI_INT, worker, TAG_STOP, MPI_COMM_WORLD);
                    ++trabalhadores_finalizados;
                }
            } else if (tag >= TAG_RESULT_BASE && tag < TAG_RESULT_BASE + altura) {
                const int y = tag - TAG_RESULT_BASE;
                // Recebe diretamente na posicao original da linha, sem copia intermediaria.
                int* destino = iteracoes.data() + static_cast<size_t>(y) * largura;
                MPI_Recv(destino, largura, MPI_INT, worker, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                ++linhas_concluidas;
                ++linhas_por_worker[worker];
            } else {
                std::cerr << "Mensagem inesperada: tag " << tag << " de rank " << worker << '\n';
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
        }

        const double tempo = MPI_Wtime() - inicio;
        if (linhas_concluidas != altura) {
            std::cerr << "Erro: linhas concluídas = " << linhas_concluidas << '\n';
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
        long long checksum = 0;
        for (int valor : iteracoes) checksum += valor;
        std::cout << "Tempo MPI: " << tempo << " segundos (" << proc_n << " processos)\n";
        std::cout << "Checksum: " << checksum << '\n';
        for (int p = 1; p < proc_n; ++p)
            std::cout << "Trabalhador " << p << ": " << linhas_por_worker[p] << " linhas\n";

        // Gravacao da imagem fora do tempo medido.
        std::vector<uint8_t> imagem(static_cast<size_t>(largura) * altura * 3);
        for (size_t i = 0; i < iteracoes.size(); ++i) {
            const int iter = iteracoes[i];
            if (iter == maxIter) {
                imagem[i * 3] = imagem[i * 3 + 1] = imagem[i * 3 + 2] = 0;
            } else {
                const double t = static_cast<double>(iter) / maxIter;
                imagem[i * 3]     = static_cast<uint8_t>(9 * (1 - t) * t * t * t * 255);
                imagem[i * 3 + 1] = static_cast<uint8_t>(15 * (1 - t) * (1 - t) * t * t * 255);
                imagem[i * 3 + 2] = static_cast<uint8_t>(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
            }
        }
        if (stbi_write_png("mandelbrot_mpi.png", largura, altura, 3, imagem.data(), largura * 3))
            std::cout << "Imagem salva em mandelbrot_mpi.png\n";
        else
            std::cerr << "Erro ao salvar a imagem.\n";
    } else {
        std::vector<int> linha(largura);
        while (true) {
            const int pedido = 1;
            MPI_Send(&pedido, 1, MPI_INT, 0, TAG_REQUEST, MPI_COMM_WORLD);

            int y;
            MPI_Recv(&y, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == TAG_STOP) break;
            if (status.MPI_TAG != TAG_WORK) {
                std::cerr << "Tag de trabalho inesperada no rank " << my_rank << '\n';
                MPI_Abort(MPI_COMM_WORLD, 4);
            }

            for (int x = 0; x < largura; ++x) {
                const double cx = minX + (maxX - minX) * x / (largura - 1.0);
                const double cy = minY + (maxY - minY) * y / (altura - 1.0);
                linha[x] = mandelbrot(cx, cy, maxIter);
            }
            // O indice y segue na tag; o coordenador usa MPI_Probe e ponteiro direto.
            MPI_Send(linha.data(), largura, MPI_INT, 0, TAG_RESULT_BASE + y, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
