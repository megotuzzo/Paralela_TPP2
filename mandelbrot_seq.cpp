#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int mandelbrot(double cx, double cy, int maxIter) {
    double zx = 0.0;
    double zy = 0.0;

    int iter = 0;

    while (zx * zx + zy * zy <= 4.0 && iter < maxIter) {
        double novoZx = zx * zx - zy * zy + cx;
        zy = 2.0 * zx * zy + cy;
        zx = novoZx;

        iter++;
    }

    return iter;
}

int main() {
    const int largura = 2000;
    const int altura = 2000;
    const int maxIter = 1000;

    const double minX = -2.0;
    const double maxX = 1.0;
    const double minY = -1.5;
    const double maxY = 1.5;

    std::vector<int> iteracoes(largura * altura);
    std::vector<uint8_t> imagem(largura * altura * 3);

    auto inicio = std::chrono::high_resolution_clock::now();

    for (int y = 0; y < altura; y++) {
        for (int x = 0; x < largura; x++) {
            double cx = minX +
                (maxX - minX) * x / (largura - 1.0);

            double cy = minY +
                (maxY - minY) * y / (altura - 1.0);

            iteracoes[y * largura + x] =
                mandelbrot(cx, cy, maxIter);
        }
    }

    auto fim = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> tempo = fim - inicio;

    long long checksum = 0;

    for (int valor : iteracoes) {
        checksum += valor;
    }

    std::cout << "Tempo de execucao: "
                << tempo.count()
                << " segundos\n";

    std::cout << "Checksum: "
                << checksum
                << '\n';

    // Geração da imagem fora da medição de tempo
    for (int y = 0; y < altura; y++) {
        for (int x = 0; x < largura; x++) {
            int iter = iteracoes[y * largura + x];

            int indice = (y * largura + x) * 3;

            if (iter == maxIter) {
                imagem[indice] = 0;
                imagem[indice + 1] = 0;
                imagem[indice + 2] = 0;
            } else {
                double t =
                    static_cast<double>(iter) / maxIter;

                imagem[indice] =
                    static_cast<uint8_t>(9 * (1 - t) * t * t * t * 255);

                imagem[indice + 1] =
                    static_cast<uint8_t>(15 * (1 - t) * (1 - t) * t * t * 255);

                imagem[indice + 2] =
                    static_cast<uint8_t>(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
            }
        }
    }

    int sucesso = stbi_write_png(
        "mandelbrot.png",
        largura,
        altura,
        3,
        imagem.data(),
        largura * 3
    );

    if (sucesso) {
        std::cout << "Imagem salva em mandelbrot.png\n";
    } else {
        std::cerr << "Erro ao salvar a imagem.\n";
    }

    return 0;
}