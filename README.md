<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: 5;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

<!-- Title -->
<h1 align="center"><b>CS529.Q11 - CÁC VẤN ĐỀ NGHIÊN CỨU VÀ ỨNG DỤNG TRONG KHOA HỌC MÁY TÍNH</b></h1>

<p align="center">
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=flat-square" alt="PyTorch"></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat-square" alt="Python"></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Research%20Only-blue.svg?style=flat-square" alt="License"></a>

</p>

## MỤC LỤC
* [Giới thiệu môn học](#gioi-thieu-mon-hoc)
* [Giảng viên hướng dẫn](#giang-vien-huong-dan)
* [Sinh viên thực hiện](#sinh-vien-thuc-hien)
* [Đồ án](#do-an)
* [Yêu cầu hệ thống](#yeu-cau-he-thong)
* [Cài đặt môi trường](#cai-dat-moi-truong)
* [Lưu ý cấu hình quan trọng](#luu-y-cau-hinh-quan-trong)
* [Dữ liệu](#du-lieu)
* [Mô hình huấn luyện sẵn](#mo-hinh-huan-luyen-san)
* [Đánh giá](#danh-gia)
* [Huấn luyện](#huan-luyen)
* [Demo](#demo)
  - [Demo dòng lệnh](#demo-dong-lenh)
  - [Web Demo](#web-demo)
* [Kaggle Notebook](#kaggle-notebook)
* [Tham khảo](#tham-khao)

## GIỚI THIỆU MÔN HỌC
<a name="gioi-thieu-mon-hoc"></a>
* **Tên môn học**: Các vấn đề nghiên cứu và ứng dụng trong Khoa học máy tính
* **Mã môn học**: CS529
* **Mã lớp**: CS529.Q11
* **Ngày bắt đầu**: 08/09/2025
* **Năm học**: 2025 - 2026

## GIẢNG VIÊN HƯỚNG DẪN
<a name="giang-vien-huong-dan"></a>
* **ThS. Đỗ Văn Tiến** - *tiendv@uit.edu.vn*

## SINH VIÊN THỰC HIỆN
<a name="sinh-vien-thuc-hien"></a>
| MSSV | Họ và tên | Github | Email |
|:----------:|:-------------------:|:----------------------------------------------------:|:-----------------------:|
| 22521587   | Trương Phúc Trường  | [Truong99zvc](https://github.com/Truong99zvc/)      | 22521587@gm.uit.edu.vn  |
| 22521571   | Võ Đình Trung       | [votrung654](https://github.com/votrung654/)         | 22521571@gm.uit.edu.vn  |

## ĐỒ ÁN
<a name="do-an"></a>
**Tên đồ án**: ĐÁNH GIÁ, TÁI LẬP VÀ MỞ RỘNG PHƯƠNG PHÁP NỘI SUY KHUNG HÌNH VIDEO DỰA TRÊN TRƯỜNG CHUYỂN ĐỘNG HAI CHIỀU CHO CÁC KỊCH BẢN CHUYỂN ĐỘNG PHỨC TẠP

**Dự án này là kết quả của quá trình tái lập (reproduce), đánh giá và mở rộng (extension) dựa trên nghiên cứu:**
> **BiM-VFI: Bidirectional Motion Field-Guided Frame Interpolation for Video with Non-uniform Motions**
> *Tác giả: Wonyong Seo, Jihyong Oh, Munchurl Kim*

Repository này chứa mã nguồn cài đặt và đánh giá của hai mô hình:
1.  **BiM-VFI (Reproduced)**: Phương pháp gốc từ paper, được tái lập hoàn chỉnh.
2.  **BiM-RNet (Proposed/Extension)**: Kiến trúc đề xuất cải tiến, tăng hiệu suất và tối ưu hóa khả năng trích xuất đặc trưng trong các kịch bản chuyển động phức tạp.

**Đóng góp của nhóm sinh viên:**
1.  **Tái lập (Reproducibility):** Huấn luyện lại mô hình BiM-VFI gốc từ đầu trên dataset Vimeo90K để kiểm chứng kết quả.
2.  **Đề xuất cải tiến (Extension - BiM-RNet):** Thiết kế và cài đặt kiến trúc **BiM-RNet**, tận dụng sức mạnh của ResNet Encoder.
3.  **Khắc phục lỗi tương thích:** Chỉnh sửa mã nguồn để chạy ổn định trên các môi trường hiện đại (Torch mới, CUDA mới) và hệ điều hành Windows.
4.  **Phát triển Ứng dụng:** Tích hợp module **Web Demo** (Flask) cho phép tương tác trực quan.
5.  **Tài liệu hóa:** Việt hóa và chi tiết hóa hướng dẫn sử dụng.

Mã nguồn cốt lõi dựa trên repo gốc của nhóm tác giả KAIST-VICLab.

## YÊU CẦU HỆ THỐNG
<a name="yeu-cau-he-thong"></a>
Để chạy mã nguồn ổn định, hệ thống cần đáp ứng các yêu cầu tối thiểu sau:

*   **Hệ điều hành**: Windows 10/11 (đã được nhóm test kỹ) hoặc Linux.
*   **GPU (Quan trọng)**:
    *   Bắt buộc **NVIDIA GPU** hỗ trợ CUDA.
    *   **VRAM**: Tối thiểu **4GB** (Test trên GTX 1650 chạy ổn định với độ phân giải thấp/trung bình). Khuyến nghị **8GB+** để inference video HD/FullHD.
    *   **Lưu ý**: Để huấn luyện (training), đặc biệt là BiM-RNet, khuyến nghị sử dụng GPU server mạnh (như Kaggle P100/T4x2 hoặc tốt hơn) vì 4GB VRAM không đủ cho batch size lớn.
*   **CUDA Driver**: Cần cài đặt NVIDIA Driver phiên bản mới nhất tương thích với CUDA 11.8 trở lên.
*   **RAM**: Tối thiểu 16GB (Quá trình xử lý video tốn nhiều RAM).

## CÀI ĐẶT MÔI TRƯỜNG
<a name="cai-dat-moi-truong"></a>

### Yêu cầu tiên quyết
Trước khi thiết lập môi trường, hãy đảm bảo đã cài đặt **Conda** trên hệ thống của mình. Có thể tải xuống và cài đặt Conda từ:
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (Khuyên dùng - nhẹ)
- [Anaconda](https://www.anaconda.com/products/distribution) (Bản đầy đủ)

### Thiết lập môi trường và cài đặt thư viện

> **Lưu ý quan trọng**: Các phiên bản thư viện trong repository này khác với repository gốc của BiM-VFI. Để đảm bảo khả năng tái lập và chạy ổn định trên các GPU phổ thông như **GTX 1650**, nhóm sử dụng phiên bản PyTorch mới nhất hỗ trợ CUDA 13.0.

```bash
conda create -n bimvfi python=3.11
conda activate bimvfi
pip install basicsr-fixed Ipython torchsummary wandb moviepy pyyaml imageio packaging tqdm opencv-python tensorboardx ptflops pyiqa lpips stlpips_pytorch dists_pytorch torch torchvision --index-url https://download.pytorch.org/whl/cu130
conda install cupy -c conda-forge
```

### Thư viện bổ sung cho Web Demo
Để chạy web demo, cần cài đặt thêm các thư viện sau:

```bash
pip install flask werkzeug pillow scikit-image
```

**Lưu ý**: `opencv-python` và `torch` đã được bao gồm trong thiết lập môi trường chính ở trên. Hãy đảm bảo đã cài đặt đầy đủ.

## LƯU Ý CẤU HÌNH QUAN TRỌNG
<a name="luu-y-cau-hinh-quan-trong"></a>

### Biến môi trường KMP_DUPLICATE_LIB_OK
Trong `main.py`, nhóm đã thêm cấu hình để giải quyết xung đột OpenMP:
```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```
Biến môi trường này xử lý lỗi "OMP: Error #15: Initializing libiomp5md.dll..." thường gặp trên Windows khi dùng chung NumPy, PyTorch và OpenCV.

### Cấu hình đường dẫn tuyệt đối

Sau khi clone repository, **bắt buộc** phải sửa đổi đường dẫn dataset và model trong các file cấu hình `cfgs/*.yaml`. Các đường dẫn tương đối mặc định thường không hoạt động chính xác trên Windows và cần đổi sang **đường dẫn tuyệt đối**.

#### Cho huấn luyện (`cfgs/bim_rnet.yaml` hoặc `cfgs/bim_vfi.yaml`):
```yaml
# Trước (sẽ KHÔNG hoạt động)
root_path: ../data/vimeo_triplet

# Sau (ví dụ - điều chỉnh theo đường dẫn thực tế)
root_path: D:/Github/BiM-RNet/data/vimeo_triplet
```

#### Cho đánh giá (`cfgs/bim_vfi_benchmark.yaml`):
```yaml
# Trước
resume: ./save/bim_rnet_train_new__400_epochs_NEW/checkpoints/model_best.pth

# Sau (ví dụ)
resume: D:/Github/BiM-RNet/save/bim_rnet_train_new__400_epochs_NEW/checkpoints/model_best.pth
root_path: D:/Github/BiM-RNet/data/vimeo_triplet
```

## DỮ LIỆU
<a name="du-lieu"></a>
### Tải xuống
Dự án sử dụng dataset Vimeo90K cho huấn luyện và kiểm thử:
> - [Vimeo90K](https://cove.thecvf.com/datasets/875)

### Chuẩn bị
Sau khi tải xuống, giải nén và đặt trong thư mục `data` theo cấu trúc dự án. Nhóm sử dụng dataset Vimeo 90K-Triplet.

## MÔ HÌNH HUẤN LUYỆN SẴN
<a name="mo-hinh-huan-luyen-san"></a>

Repository này bao gồm các checkpoint:

### 1. Pretrained model gốc (BiM-VFI)
- **Đường dẫn**: `pretrained/bim_vfi.pth`
- **Mô tả**: Pretrained model gốc từ bài báo để so sánh và làm baseline.

### 2. BiM-RNet (Proposed Model)
- **Đường dẫn**: `save/bim_rnet_train_new__400_epochs_NEW/checkpoints/model_best.pth`
- **Mô tả**: Model cải tiến của nhóm, đã được huấn luyện trên Vimeo90K.

## ĐÁNH GIÁ
<a name="danh-gia"></a>
Việc đánh giá được cấu hình thông qua file `cfgs/bim_vfi_benchmark.yaml`.

*   **Config**: Điều chỉnh `name` (tên dataset) và `root_path` (đường dẫn tuyệt đối tới data) trong file cấu hình.
*   **Model**: Đảm bảo tham số `resume` trỏ đến đúng checkpoint cần đánh giá (BiM-VFI gốc hoặc BiM-RNet).

Chạy lệnh:
```bash
python main.py --cfg cfgs/bim_vfi_benchmark.yaml
```

## HUẤN LUYỆN
<a name="huan-luyen"></a>
Dự án hỗ trợ huấn luyện cả model gốc BiM-VFI và model đề xuất BiM-RNet.

### 1. Huấn luyện BiM-RNet (Mô hình cải tiến)
Sử dụng cấu hình `cfgs/bim_rnet.yaml`.

**Huấn luyện 1 GPU:**
```bash
python main.py --cfg cfgs/bim_rnet.yaml
```

**Huấn luyện đa GPU (Ví dụ 4 GPU: 0, 1, 2, 3):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 main.py --cfg cfgs/bim_rnet.yaml
```

### 2. Huấn luyện BiM-VFI (Mô hình gốc / Reproduce)
Sử dụng cấu hình `cfgs/bim_vfi.yaml`.

**Huấn luyện 1 GPU:**
```bash
python main.py --cfg cfgs/bim_vfi.yaml
```

**Huấn luyện đa GPU:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 main.py --cfg cfgs/bim_vfi.yaml
```

Để sử dụng **WandB** để theo dõi, điền thông tin vào `wandb.yaml` và thêm cờ `-w`.

## DEMO
<a name="demo"></a>
### Demo dòng lệnh
<a name="demo-dong-lenh"></a>
Các video tùy chỉnh ở định dạng nhiều ảnh hoặc video có thể được nội suy như sau.

Đầu tiên, thiết lập thư mục gốc demo như sau:
  - video1.mp4 
  - video2.mp4
  - video3
    - img0.png
    - img1.png
    - ...

Sau đó, thay thế `root_path` trong `cfgs/bim_vfi_demo.yaml` thành đường dẫn dữ liệu mong muốn, và chạy:
```bash
python main.py --cfg cfgs/bim_vfi_demo.yaml
```

### Web Demo
<a name="web-demo"></a>
Dự án bao gồm một giao diện demo dựa trên web để dễ dàng nội suy khung hình video, hỗ trợ tương tác trực quan.

1.  **Điều hướng đến thư mục web_demo**:
    ```bash
    cd web_demo
    ```
2.  **Chạy ứng dụng Flask**:
    ```bash
    python app.py
    ```
3.  **Truy cập giao diện web**:
    Mở trình duyệt tại `http://localhost:5000`

**Tính năng:**
*   **Nội suy cặp ảnh**: Tải lên 2 ảnh và tạo khung hình giữa.
*   **Nội suy video**: Tăng frame rate cho video tải lên.
*   **Nội suy chuỗi khung hình**: Xử lý mượt mà chuỗi nhiều ảnh.
*   **Lựa chọn Model**: Tùy chọn chạy BiM-VFI (Gốc) hoặc BiM-RNet (Cải tiến).
*   **Thống kê**: Xem thời gian xử lý thực tế.

## KAGGLE NOTEBOOK
<a name="kaggle-notebook"></a>

Đối với người dùng **có GPU không tương thích, không có GPU**, hoặc muốn huấn luyện/đánh giá trên cloud, nhóm cung cấp Kaggle notebook mẫu cho từng mô hình:

> **1. Notebook BiM-VFI (Reproduce)**: [https://www.kaggle.com/code/truong9/bim-vfi?scriptVersionId=289083440](https://www.kaggle.com/code/truong9/bim-vfi?scriptVersionId=289083440)
>
> **2. Notebook BiM-RNet (Proposed)**: [https://www.kaggle.com/code/gtekx9/bim-rnet](https://www.kaggle.com/code/gtekx9/bim-rnet)

**Lưu ý**: Các notebook đã được cấu hình sẵn môi trường và dataset (Vimeo90K) trên Kaggle, người dùng chỉ cần copy và run (sử dụng GPU P100 hoặc T4x2).

## THAM KHẢO
<a name="tham-khao"></a>
Đồ án sử dụng và mở rộng mã nguồn từ repository [KAIST-VICLab/BiM-VFI](https://github.com/KAIST-VICLab/BiM-VFI). Nhóm thực hiện xin gửi lời cảm ơn chân thành đến các tác giả.

Mã nguồn được tham khảo, tinh chỉnh và phát triển nhằm mục đích nghiên cứu khoa học và học tập.

Nếu sử dụng mã nguồn này cho nghiên cứu, vui lòng trích dẫn bài báo gốc:

```bibtex
@inproceedings{Seo_2025_CVPR,
  title={BiM-VFI: Bidirectional Motion Field-Guided Frame Interpolation for Video with Non-uniform Motions},
  author={Seo, Wonyong and Oh, Jihyong and Kim, Munchurl},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={7244--7253},
  year={2025}
}
```

