#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: <SEU NOME AQUI>
Disciplina: Computação Gráfica
Data: <DATA DE INÍCIO DA IMPLEMENTAÇÃO>
"""

import time         # Para operações com tempo
import gpu          # Simula os recursos de uma GPU
import math         # Funções matemáticas
import numpy as np  # Biblioteca do Numpy

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante

    matriz_modelo = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]

    pilha_matrizes = []

    posicao_camera = [0, 0, 10]
    orientacao_camera = [0, 0, 1, 0]
    campo_visao = math.pi / 4

    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far

    @staticmethod
    def _mat_mult(A, B):
        """Multiplica duas matrizes 4x4."""
        resultado = [[0, 0, 0, 0] for _ in range(4)]
        for linha in range(4):
            for coluna in range(4):
                soma = 0
                for k in range(4):
                    soma += A[linha][k] * B[k][coluna]
                resultado[linha][coluna] = soma
        return resultado

    @staticmethod
    def _mat_vec_mult(matriz, vetor):
        """Multiplica uma matriz 4x4 por um vetor homogêneo [x, y, z, 1]."""
        resultado = [0, 0, 0, 0]
        for linha in range(4):
            for coluna in range(4):
                resultado[linha] += matriz[linha][coluna] * vetor[coluna]
        return resultado

    @staticmethod
    def _matriz_rotacao(eixo_x, eixo_y, eixo_z, angulo):
        """Monta a matriz 4x4 de rotação a partir de um eixo (x, y, z) e um ângulo."""
        tamanho = math.sqrt(eixo_x ** 2 + eixo_y ** 2 + eixo_z ** 2)

        if tamanho == 0:
            return [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]

        eixo_x /= tamanho
        eixo_y /= tamanho
        eixo_z /= tamanho

        c = math.cos(angulo)
        s = math.sin(angulo)
        t = 1 - c

        return [
            [t * eixo_x * eixo_x + c, t * eixo_x * eixo_y - s * eixo_z, t * eixo_x * eixo_z + s * eixo_y, 0],
            [t * eixo_x * eixo_y + s * eixo_z, t * eixo_y * eixo_y + c, t * eixo_y * eixo_z - s * eixo_x, 0],
            [t * eixo_x * eixo_z - s * eixo_y, t * eixo_y * eixo_z + s * eixo_x, t * eixo_z * eixo_z + c, 0],
            [0, 0, 0, 1]
        ]

    @staticmethod
    def _transformar_ponto(x, y, z):
        """Aplica modelo -> câmera -> projeção perspectiva -> tela.

        Retorna [x_tela, y_tela] ou None se o ponto ficar atrás da câmera
        (não deve ser desenhado)."""
        ponto = GL._mat_vec_mult(GL.matriz_modelo, [x, y, z, 1])

        # Translação para a posição da câmera.
        ponto[0] -= GL.posicao_camera[0]
        ponto[1] -= GL.posicao_camera[1]
        ponto[2] -= GL.posicao_camera[2]

        # Rotação inversa da câmera.
        eixo_x, eixo_y, eixo_z, angulo = GL.orientacao_camera
        matriz_rotacao_inversa = GL._matriz_rotacao(eixo_x, eixo_y, eixo_z, -angulo)
        ponto = GL._mat_vec_mult(matriz_rotacao_inversa, ponto)

        # Projeção perspectiva.
        z_camera = ponto[2]

        if z_camera >= -0.1:
            return None

        aspecto = GL.width / GL.height
        f = 1 / math.tan(GL.campo_visao / 2)

        x_projetado = (ponto[0] * f / aspecto) / -z_camera
        y_projetado = (ponto[1] * f) / -z_camera

        x_tela = round((x_projetado + 1) * GL.width / 2)
        y_tela = round((1 - y_projetado) * GL.height / 2)

        return [x_tela, y_tela]

    @staticmethod
    def _lado(x, y, x0, y0, x1, y1):
        """Calcula de que lado da reta (x0,y0)-(x1,y1) o ponto (x,y) está."""
        return (y1 - y0) * x - (x1 - x0) * y + y0 * (x1 - x0) - x0 * (y1 - y0)

    @staticmethod
    def _dentro_triangulo(x, y, P0, P1, P2):
        """Testa se o pixel (x,y) está dentro do triângulo P0-P1-P2."""
        L0 = GL._lado(x, y, P0[0], P0[1], P1[0], P1[1])
        L1 = GL._lado(x, y, P1[0], P1[1], P2[0], P2[1])
        L2 = GL._lado(x, y, P2[0], P2[1], P0[0], P0[1])

        return (
            (L0 >= 0 and L1 >= 0 and L2 >= 0) or
            (L0 <= 0 and L1 <= 0 and L2 <= 0)
        )

    @staticmethod
    def _rasterizar_triangulo(p0, p1, p2, cor):
        """Percorre a bounding box (já recortada pela tela) e pinta os pixels internos."""
        min_x = max(min(p0[0], p1[0], p2[0]), 0)
        max_x = min(max(p0[0], p1[0], p2[0]), GL.width - 1)
        min_y = max(min(p0[1], p1[1], p2[1]), 0)
        max_y = min(max(p0[1], p1[1], p2[1]), GL.height - 1)

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if GL._dentro_triangulo(x, y, p0, p1, p2):
                    gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, cor)

    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polypoint2D
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é a
        # coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista e assuma que sempre vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polypoint2D
        # você pode assumir inicialmente o desenho dos pontos com a cor emissiva (emissiveColor).
        cor = [round(componente * 255)
            for componente in colors["emissiveColor"]]

        for i in range(0, len(point), 2):
            x = round(point[i])
            y = round(point[i + 1])

            if 0 <= x < GL.width and 0 <= y < GL.height:
                gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, cor)
        
    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polyline2D
        # Nessa função você receberá os pontos de uma linha no parâmetro lineSegments, esses
        # pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o valor da
        # coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é
        # a coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista. A quantidade mínima de pontos são 2 (4 valores), porém a
        # função pode receber mais pontos para desenhar vários segmentos. Assuma que sempre
        # vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polyline2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).

        cor = [round(componente * 255)
            for componente in colors["emissiveColor"]]

        for i in range(0, len(lineSegments) - 2, 2):
            x0 = lineSegments[i]
            y0 = lineSegments[i + 1]
            x1 = lineSegments[i + 2]
            y1 = lineSegments[i + 3]

            dx = x1 - x0
            dy = y1 - y0

            if abs(dx) >= abs(dy):
                if x0 > x1:
                    x0, x1 = x1, x0
                    y0, y1 = y1, y0

                if x1 != x0:
                    s = (y1 - y0) / (x1 - x0)
                else:
                    s = 0

                y = y0

                for x in range(math.floor(x0), math.floor(x1) + 1):
                    pixel_y = math.floor(y)
                    if 0 <= x < GL.width and 0 <= pixel_y < GL.height:
                        gpu.GPU.draw_pixel([x, pixel_y], gpu.GPU.RGB8, cor)
                    y += s

            else:
                if y0 > y1:
                    x0, x1 = x1, x0
                    y0, y1 = y1, y0

                s = (x1 - x0) / (y1 - y0)
                x = x0

                for y in range(math.floor(y0), math.floor(y1) + 1):
                    pixel_x = math.floor(x)
                    if 0 <= pixel_x < GL.width and 0 <= y < GL.height:
                        gpu.GPU.draw_pixel([pixel_x, y], gpu.GPU.RGB8, cor)
                    x += s
                

    @staticmethod
    def circle2D(radius, colors):
        """Função usada para renderizar Circle2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Circle2D
        # Nessa função você receberá um valor de raio e deverá desenhar o contorno de
        # um círculo.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Circle2D
        # você pode assumir o desenho das linhas com a cor emissiva (emissiveColor).
        cor = [round(componente * 255)
                    for componente in colors["emissiveColor"]]

        def inside(x, y, cx, cy, r):
            distancia = (x - cx)**2 + (y - cy)**2
            return abs(distancia - r**2) <= r

        x_centro = 0
        y_centro = 0

        for y in range(GL.height):
            for x in range(GL.width):
                if inside(x, y, x_centro, y_centro, radius):
                    if 0 <= x < GL.width and 0 <= y < GL.height:
                        gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, cor)

    @staticmethod
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#TriangleSet2D
        # Nessa função você receberá os vertices de um triângulo no parâmetro vertices,
        # esses pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o
        # valor da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto.
        # Já point[2] é a coordenada x do segundo ponto e assim por diante. Assuma que a
        # quantidade de pontos é sempre multiplo de 3, ou seja, 6 valores ou 12 valores, etc.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o TriangleSet2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).
        cor = [round(componente * 255)
            for componente in colors["emissiveColor"]]

        for i in range(0, len(vertices), 6):
            P0 = (vertices[i],     vertices[i + 1])
            P1 = (vertices[i + 2], vertices[i + 3])
            P2 = (vertices[i + 4], vertices[i + 5])

            GL._rasterizar_triangulo(P0, P1, P2, cor)

    @staticmethod
    def triangleSet(point, colors):
        """Função usada para renderizar TriangleSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleSet
        # No TriangleSet os triângulos são informados individualmente, assim os três
        # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
        # triângulo, e assim por diante.

        cor = [round(componente * 255)
            for componente in colors["emissiveColor"]]

        # Cada triângulo possui 9 valores (3 vértices x, y, z)
        for i in range(0, len(point), 9):

            p0 = GL._transformar_ponto(point[i], point[i + 1], point[i + 2])
            p1 = GL._transformar_ponto(point[i + 3], point[i + 4], point[i + 5])
            p2 = GL._transformar_ponto(point[i + 6], point[i + 7], point[i + 8])

            if p0 is not None and p1 is not None and p2 is not None:
                GL._rasterizar_triangulo(p0, p1, p2, cor)

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        GL.posicao_camera = position
        GL.orientacao_camera = orientation
        GL.campo_visao = fieldOfView

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # Empilha a matriz do PAI antes de calcular a matriz local, para podermos
        # restaurá-la depois em transform_out().
        GL.pilha_matrizes.append(GL.matriz_modelo)

        tx, ty, tz = [0, 0, 0] if not translation else translation
        sx, sy, sz = [1, 1, 1] if not scale else scale

        matriz_translacao = [
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ]

        matriz_escala = [
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1]
        ]

        if rotation:
            eixo_x, eixo_y, eixo_z, angulo = rotation
            matriz_rotacao = GL._matriz_rotacao(eixo_x, eixo_y, eixo_z, angulo)
        else:
            matriz_rotacao = [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]

        # Ordem padrão X3D para a matriz local: Translação * Rotação * Escala
        # (escala primeiro, depois rotaciona, depois translada).
        matriz_local = GL._mat_mult(GL._mat_mult(matriz_translacao, matriz_rotacao), matriz_escala)

        # Combina com a matriz do pai: mundo = pai * local
        GL.matriz_modelo = GL._mat_mult(GL.matriz_modelo, matriz_local)

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # Ao sair do nó Transform, recuperamos a matriz do pai que empilhamos em
        # transform_in().
        GL.matriz_modelo = GL.pilha_matrizes.pop()

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleStripSet
        # point é uma lista de vértices x, y, z. stripCount informa quantos vértices
        # (não quantos triângulos) cada tira consome, na ordem. Dentro de uma tira,
        # o triângulo j é formado pelos vértices (j, j+1, j+2), mas para manter a
        # orientação (winding) consistente, invertemos a ordem dos dois primeiros
        # vértices quando j é ímpar.

        cor = [round(componente * 255)
            for componente in colors["emissiveColor"]]

        pontos3D = [
            (point[i], point[i + 1], point[i + 2])
            for i in range(0, len(point), 3)
        ]

        indice = 0
        for count in stripCount:
            tira = pontos3D[indice:indice + count]
            indice += count

            pontos_tela = [GL._transformar_ponto(*p) for p in tira]

            for j in range(len(pontos_tela) - 2):
                if j % 2 == 0:
                    p0, p1, p2 = pontos_tela[j], pontos_tela[j + 1], pontos_tela[j + 2]
                else:
                    p0, p1, p2 = pontos_tela[j + 1], pontos_tela[j], pontos_tela[j + 2]

                if p0 is not None and p1 is not None and p2 is not None:
                    GL._rasterizar_triangulo(p0, p1, p2, cor)

    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#IndexedTriangleStripSet
        # point contém os vértices x, y, z. index é uma lista de índices para esses
        # vértices, onde -1 marca o fim de uma tira (pode haver mais de uma tira na
        # mesma chamada). A regra de montagem dos triângulos dentro de cada tira é a
        # mesma do triangleStripSet.

        cor = [round(componente * 255)
            for componente in colors["emissiveColor"]]

        pontos3D = [
            (point[i], point[i + 1], point[i + 2])
            for i in range(0, len(point), 3)
        ]

        # Separa a lista de índices em tiras, usando -1 como delimitador.
        tiras = []
        tira_atual = []
        for idx in index:
            if idx == -1:
                if tira_atual:
                    tiras.append(tira_atual)
                tira_atual = []
            else:
                tira_atual.append(idx)
        if tira_atual:
            tiras.append(tira_atual)

        for tira in tiras:
            pontos_tela = [GL._transformar_ponto(*pontos3D[i]) for i in tira]

            for j in range(len(pontos_tela) - 2):
                if j % 2 == 0:
                    p0, p1, p2 = pontos_tela[j], pontos_tela[j + 1], pontos_tela[j + 2]
                else:
                    p0, p1, p2 = pontos_tela[j + 1], pontos_tela[j], pontos_tela[j + 2]

                if p0 is not None and p1 is not None and p2 is not None:
                    GL._rasterizar_triangulo(p0, p1, p2, cor)

    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                       texCoord, texCoordIndex, colors, current_texture):
        """Função usada para renderizar IndexedFaceSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#IndexedFaceSet
        # coord contém os vértices x, y, z. coordIndex é uma lista de índices desses
        # vértices, onde -1 marca o fim de uma face (a face pode ter mais de 3
        # vértices). Cada face é triangulada em leque a partir do primeiro vértice:
        # (v0, v1, v2), (v0, v2, v3), (v0, v3, v4), ...

        cor = [round(componente * 255)
            for componente in colors["emissiveColor"]]

        if not coord or not coordIndex:
            return

        pontos3D = [
            (coord[i], coord[i + 1], coord[i + 2])
            for i in range(0, len(coord), 3)
        ]

        # Separa coordIndex em faces, usando -1 como delimitador.
        faces = []
        face_atual = []
        for idx in coordIndex:
            if idx == -1:
                if face_atual:
                    faces.append(face_atual)
                face_atual = []
            else:
                face_atual.append(idx)
        if face_atual:
            faces.append(face_atual)

        for face in faces:
            if len(face) < 3:
                continue

            v0 = face[0]
            p0 = GL._transformar_ponto(*pontos3D[v0])

            for k in range(1, len(face) - 1):
                v1 = face[k]
                v2 = face[k + 1]

                p1 = GL._transformar_ponto(*pontos3D[v1])
                p2 = GL._transformar_ponto(*pontos3D[v2])

                if p0 is not None and p1 is not None and p2 is not None:
                    GL._rasterizar_triangulo(p0, p1, p2, cor)

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Box
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size)) # imprime no terminal pontos
        print("Box : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Sphere
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Sphere : radius = {0}".format(radius)) # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def cone(bottomRadius, height, colors):
        """Função usada para renderizar Cones."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cone
        # A função cone é usada para desenhar cones na cena. O cone é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento bottomRadius especifica o
        # raio da base do cone e o argumento height especifica a altura do cone.
        # O cone é alinhado com o eixo Y local. O cone é fechado por padrão na base.
        # Para desenha esse cone você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cone : bottomRadius = {0}".format(bottomRadius)) # imprime no terminal o raio da base do cone
        print("Cone : height = {0}".format(height)) # imprime no terminal a altura do cone
        print("Cone : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def cylinder(radius, height, colors):
        """Função usada para renderizar Cilindros."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cylinder
        # A função cylinder é usada para desenhar cilindros na cena. O cilindro é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da base do cilindro e o argumento height especifica a altura do cilindro.
        # O cilindro é alinhado com o eixo Y local. O cilindro é fechado por padrão em ambas as extremidades.
        # Para desenha esse cilindro você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cylinder : radius = {0}".format(radius)) # imprime no terminal o raio do cilindro
        print("Cylinder : height = {0}".format(height)) # imprime no terminal a altura do cilindro
        print("Cylinder : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/navigation.html#NavigationInfo
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("NavigationInfo : headlight = {0}".format(headlight)) # imprime no terminal

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#DirectionalLight
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("DirectionalLight : ambientIntensity = {0}".format(ambientIntensity))
        print("DirectionalLight : color = {0}".format(color)) # imprime no terminal
        print("DirectionalLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("DirectionalLight : direction = {0}".format(direction)) # imprime no terminal

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#PointLight
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color)) # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("PointLight : location = {0}".format(location)) # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/environmentalEffects.html#Fog
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color)) # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/time.html#TimeSensor
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("TimeSensor : cycleInterval = {0}".format(cycleInterval)) # imprime no terminal
        print("TimeSensor : loop = {0}".format(loop))

        # Esse método já está implementado para os alunos como exemplo
        epoch = time.time()  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#SplinePositionInterpolator
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        print("SplinePositionInterpolator : key = {0}".format(key)) # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0.0, 0.0, 0.0]
        
        return value_changed

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#OrientationInterpolator
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        print("OrientationInterpolator : key = {0}".format(key)) # imprime no terminal
        print("OrientationInterpolator : keyValue = {0}".format(keyValue))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0, 0, 1, 0]

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
