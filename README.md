import cv2
import dlib

# Carregar o detector de rostos
detector = dlib.get_frontal_face_detector()
# Carregar o modelo de predição de marcos faciais
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def reconhecer_rosto(imagem):
    # Carregar a imagem
    img = cv2.imread(imagem)
    # Converter a imagem para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detectar rostos na imagem
    faces = detector(gray)

    for face in faces:
        # Obter os marcos faciais
        landmarks = predictor(gray, face)
        # Desenhar retângulo ao redor do rosto
        cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        # Desenhar círculos nos marcos faciais
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(img, (x, y), 1, (255, 0, 0), -1)

    # Salvar a imagem com os rostos detectados
    cv2.imwrite("output.jpg", img)
    return "output.jpg"

# Testar a função com uma imagem de entrada
reconhecer_rosto('imagem_de_teste.jpg')
