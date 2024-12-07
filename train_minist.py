import os
import numpy as np
from PNGProcessor import PNGProcessor  # PNGProcessor 클래스가 포함된 파일을 임포트
from DataPreProcessor import DataPreProcessor
from MLPForMINIST import MLPForMINIST  # MLP 모델 클래스

# 데이터 전처리 객체 초기화
data_processor = DataPreProcessor()

def load_mnist_dataset(base_path):
    """
    주어진 경로에서 폴더별로 PNG 파일을 읽어와 리스트에 저장합니다.
    base_path: MNIST 데이터셋이 포함된 디렉토리 경로
    return: (images, labels) 이미지 데이터와 레이블 리스트
    """
    png_processor = PNGProcessor()
    images = []
    labels = []

    for label in range(10):  # 0부터 9까지 폴더 순회
        folder_path = os.path.join(base_path, str(label))
        if not os.path.isdir(folder_path):
            print(f"Warning: {folder_path} is not a valid directory.")
            continue

        print(f"Loading images from folder: {folder_path}")
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".png"):
                file_path = os.path.join(folder_path, file_name)
                pixels = png_processor.open_png(file_path)

                if pixels is not None:  # 유효한 데이터 추가
                    images.append(data_processor.flatten(pixels))
                    labels.append(label)

    return images, labels

# 데이터 로드
base_path = "test"  # MNIST 데이터셋 경로
images, labels = load_mnist_dataset(base_path)

# 데이터 전처리
print(f"Total images loaded: {len(images)}")
print(f"Total labels loaded: {len(labels)}")

# 라벨 원-핫 인코딩
encoded_labels = data_processor.encode(labels)
#print(f"First 3 encoded labels: {encoded_labels[:3]}")

# 이미지 정규화
scaled_images = data_processor.scale_data(images)
#print(f"First scaled image (flattened): {scaled_images[0]}")

# 학습/테스트 데이터 나누기
features, targets = scaled_images, encoded_labels
X_train, y_train, X_test, y_test = data_processor.shuffle_train_test_data(
    features, targets, train_ratio=0.8, seed=42
)
print(f"Data split: {len(X_train)} train samples, {len(X_test)} test samples")

# NumPy 배열로 변환
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# MLP 모델 생성
input_size = 28 * 28  # MNIST 이미지 크기 (28x28)
hidden_size = 128     # 은닉층 뉴런 개수
output_size = 10      # 출력 뉴런 개수 (0~9 숫자)
learning_rate = 0.2   # 학습률

mlp_model = MLPForMINIST(input_size, hidden_size, output_size, learning_rate)

# 모델 학습
train_losses, test_losses = mlp_model.fit(X_train, y_train, X_test, y_test, epoch=500, patience=10)

# 모델 평가
y_pred = mlp_model.predict(X_test)
accuracy = np.mean(y_pred == np.argmax(y_test, axis=1))
print(f"score: {accuracy * 100:.2f}")


###########
# 학습 완료된 모델 파라미터 저장
save_folder = "mlp_model_params"
mlp_model.save_model(save_folder)

print(f"parameters saved ({save_folder})")
