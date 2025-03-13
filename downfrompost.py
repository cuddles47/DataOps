import psycopg2
import csv
import os

# Cấu hình kết nối PostgreSQL
conn = psycopg2.connect(
    dbname='airflow_db',
    user='airflow_user',
    password='airflow_pass',
    host='host.docker.internal',
    port=5432
)

# Thư mục để lưu dữ liệu dưới dạng file CSV
output_folder = '/opt/airflow/scripts/python/data'  # Thay bằng đường dẫn mong muốn

# Tạo thư mục nếu chưa tồn tại
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Tên bảng bạn muốn lấy dữ liệu
table_name = 'excel_data'  # Thay bằng tên bảng mong muốn

def export_table_to_csv():
    try:
        # Kết nối đến PostgreSQL
        cursor = conn.cursor()
        print("✅ Kết nối thành công đến PostgreSQL")

        # Kiểm tra xem bảng có tồn tại không
        cursor.execute(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = '{table_name}'
            );
        """)
        table_exists = cursor.fetchone()[0]

        if not table_exists:
            print(f"❌ Bảng '{table_name}' không tồn tại.")
            return

        # Lấy dữ liệu từ bảng
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        if len(rows) == 0:
            print(f"⚠️ Bảng '{table_name}' không có dữ liệu.")
            return

        # Lấy tên các cột
        colnames = [desc[0] for desc in cursor.description]

        # Đường dẫn file CSV
        csv_file_path = os.path.join(output_folder, f"{table_name}.csv")

        # Ghi dữ liệu vào file CSV
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(colnames)  # Ghi tên cột
            csvwriter.writerows(rows)  # Ghi dữ liệu
            print(f"✅ Đã lưu dữ liệu từ bảng '{table_name}' vào file: {csv_file_path}")

    except Exception as e:
        print(f"❌ Lỗi: {e}")
    finally:
        cursor.close()
        conn.close()
        print("🔌 Đã ngắt kết nối với PostgreSQL.")

# Gọi hàm để xuất dữ liệu từ bảng cụ thể
export_table_to_csv()