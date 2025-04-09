from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import os

# Hàm trích xuất dữ liệu từ các tệp CSV và lưu dưới dạng tệp Parquet
def extract_data():
    data_path = "/path/to/dataset/"  # Đường dẫn tới thư mục chứa dữ liệu
    files = [f for f in os.listdir(data_path) if f.endswith(".csv")]  # Lấy danh sách tất cả các tệp CSV
    dataframes = [pd.read_csv(os.path.join(data_path, file), low_memory=False) for file in files]  # Đọc từng tệp CSV vào DataFrame
    df = pd.concat(dataframes, axis=0)  # Gộp tất cả DataFrame thành một
    df.to_parquet(os.path.join(data_path, "raw_data.parquet"), index=False)  # Lưu dữ liệu dưới dạng Parquet

# Hàm làm sạch dữ liệu bằng cách loại bỏ các bản ghi trùng lặp
def transform_data():
    data_path = "/path/to/dataset/"  # Đường dẫn tới thư mục chứa dữ liệu
    df = pd.read_parquet(os.path.join(data_path, "raw_data.parquet"))  # Đọc dữ liệu từ tệp Parquet
    df = df.drop_duplicates()  # Loại bỏ các dòng trùng lặp
    df = df.T.drop_duplicates().T  # Loại bỏ các cột trùng lặp
    df.to_parquet(os.path.join(data_path, "cleaned_data.parquet"), index=False)  # Lưu dữ liệu đã làm sạch dưới dạng Parquet

# Hàm giả lập việc lưu dữ liệu đã làm sạch
def save_cleaned_data():
    print("Dữ liệu đã được làm sạch và lưu thành công!")

# Định nghĩa các tham số mặc định cho DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,  # Không phụ thuộc vào lần chạy trước
    'start_date': datetime(2024, 3, 13),  # Ngày bắt đầu chạy DAG
    'retries': 1,  # Số lần thử lại nếu tác vụ thất bại
    'retry_delay': timedelta(minutes=5),  # Thời gian chờ giữa các lần thử lại
}

# Khởi tạo DAG
dag = DAG(
    'house_prices_dag',  # Tên DAG
    default_args=default_args,
    description='DAG xử lý và làm sạch dữ liệu giá nhà',
    schedule_interval='@daily',  # Lên lịch chạy DAG mỗi ngày
    catchup=False,  # Không chạy lại các DAG bị bỏ lỡ
)

# Định nghĩa các tác vụ trong DAG
extract_task = PythonOperator(
    task_id='extract_data',  # Tác vụ trích xuất dữ liệu
    python_callable=extract_data,
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform_data',  # Tác vụ làm sạch dữ liệu
    python_callable=transform_data,
    dag=dag,
)

save_task = PythonOperator(
    task_id='save_cleaned_data',  # Tác vụ lưu dữ liệu đã làm sạch
    python_callable=save_cleaned_data,
    dag=dag,
)

# Xác định trình tự thực thi của các tác vụ
extract_task >> transform_task >> save_task
