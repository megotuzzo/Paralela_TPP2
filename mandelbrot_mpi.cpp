#include <iostream>
#include <vector>
#include <cstdint>
#include <mpi.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define TAG_WORK   1
#define TAG_RESULT 2
#define TAG_STOP   3

// função de matemática do T1
int mandelbrot(double cx, double cy, int maxIter) {
    double zx = 0.0, zy = 0.0;
    int iter = 0;
    while (zx * zx + zy * zy <= 4.0 && iter < maxIter) {
        double novoZx = zx * zx - zy * zy + cx;
        zy = 2.0 * zx * zy + cy;
        zx = novoZx;
        iter++;
    }
    return iter;
}

int main(int argc, char* argv[]) {
    int my_rank, proc_n;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_n);

    if (proc_n < 2) {
        if (my_rank == 0) std::cerr << "Precisa de pelo menos 2 processos!\n";
        MPI_Finalize();
        return 1;
    }

    // Config da Imagem
    int largura = 2000;
    int altura  = 2000;
    int maxIter = 5000;
    double minX = -2.0, maxX = 1.0;
    double minY = -1.5, maxY = 1.5;

    // payload: posição [0] = índice da linha Y, posições [1..largura] = pixels
    // +1 pra guardar a posição desta linha
    std::vector<int> payload(largura + 1);

    if (my_rank == 0) {
        //laço coordenador
        std::vector<int> iteracoes(largura * altura);
        double t_start = MPI_Wtime();

        int linhas_enviadas = 0;
        int linhas_concluidas = 0;

        // distribui uma linha inteira pra cada trabalhador
        for (int p = 1; p < proc_n && linhas_enviadas < altura; p++) {
            MPI_Send(&linhas_enviadas, 1, MPI_INT, p, TAG_WORK, MPI_COMM_WORLD);
            linhas_enviadas++;
        }

        // recebe resultado, cola na imagem, manda próxima linha pro mesmo rabalhador
        while (linhas_concluidas < altura) {
            MPI_Recv(payload.data(), largura + 1, MPI_INT, MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);
            int worker = status.MPI_SOURCE;
            int y = payload[0];

            // Cola a linha na mesma posiçao que foi retirada (y)
            for (int x = 0; x < largura; x++) {
                iteracoes[y * largura + x] = payload[x + 1];
            }
            linhas_concluidas++;

            // Manda mais trabalho ou manda parar
            if (linhas_enviadas < altura) {
                MPI_Send(&linhas_enviadas, 1, MPI_INT, worker, TAG_WORK, MPI_COMM_WORLD);
                linhas_enviadas++;
            } else { //se todas linhas ja foram enviadas pra algum trabalhador
                int dummy = -1; //'mata' o trabalahador
                MPI_Send(&dummy, 1, MPI_INT, worker, TAG_STOP, MPI_COMM_WORLD);
            }
        }

        double t_end = MPI_Wtime();
        std::cout << "Tempo: " << (t_end - t_start) << " segundos (" << proc_n << " processos)" << std::endl;

        // gera imagem PNG (igual ao T1)
        std::vector<uint8_t> imagem(largura * altura * 3);
        for (int i = 0; i < largura * altura; i++) {
            int iter = iteracoes[i];
            if (iter == maxIter) {
                imagem[i * 3]     = 0;
                imagem[i * 3 + 1] = 0;
                imagem[i * 3 + 2] = 0;
            } else {
                imagem[i * 3]     = (iter * 9) % 256;
                imagem[i * 3 + 1] = (iter * 5) % 256;
                imagem[i * 3 + 2] = (iter * 15) % 256;
            }
        }
        stbi_write_png("mandelbrot_mpi.png", largura, altura, 3, imagem.data(), largura * 3);
        std::cout << "Imagem salva em mandelbrot_mpi.png" << std::endl;

    } else { //laço trabalhor
        while (true) {
            //recebe a linha e separa a sua posiçao original 
            int y;
            MPI_Recv(&y, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_STOP) break;


            payload[0] = y;
            for (int x = 0; x < largura; x++) {
                double cx = minX + x * (maxX - minX) / largura;
                double cy = minY + y * (maxY - minY) / altura;
                payload[x + 1] = mandelbrot(cx, cy, maxIter);
            }

            MPI_Send(payload.data(), largura + 1, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
