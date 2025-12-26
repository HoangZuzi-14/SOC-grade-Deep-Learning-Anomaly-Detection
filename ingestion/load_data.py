import shutil
from pathlib import Path


def load_log_data():
    log_folder = Path("..") / "data" / "raw"
    output_file = Path("full_log.log")

    if not log_folder.exists():
        print("Lỗi: Thư mục nguồn không tồn tại.")
        return "ERROR"

    source_mtime = log_folder.stat().st_mtime

    latest_log_mtime = max([f.stat().st_mtime for f in log_folder.glob("*.log")], default=0)
    newest_change = max(source_mtime, latest_log_mtime)

    if output_file.exists():
        if output_file.stat().st_mtime >= newest_change:
            print("Dữ liệu nguồn không có thay đổi. Bỏ qua nạp lại.")
            return "NO_CHANGES"
        else:
            print("Phát hiện thay đổi ở dữ liệu nguồn. Đang cập nhật lại...")
    else:
        print("Khởi tạo nạp dữ liệu lần đầu...")

    try:
        with open(output_file, "w", encoding="utf-8") as f_out:
            for path in sorted(log_folder.glob("*.log")):
                with open(path, "r", encoding="utf-8") as f_in:
                    shutil.copyfileobj(f_in, f_out)
                    f_out.write("\n")

        print("Cập nhật dữ liệu thành công.")
        return "OK"
    except Exception as e:
        print(f"Lỗi: {e}")
        return "ERROR"

