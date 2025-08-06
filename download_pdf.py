import requests

def download_pdf(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP request errors

        # Check if the response is a PDF
        if 'application/pdf' in response.headers.get('Content-Type', ''):
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print(f"✅ PDF downloaded successfully and saved to: {save_path}")
        else:
            print("❌ The URL does not point to a valid PDF file.")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading PDF: {e}")

# === Example usage ===
pdf_url = "https://ebookcentral.proquest.com/lib/uci/reader.action?docID=6925615&c=UERG&ppg=30"
save_file_path = "sample.pdf"

download_pdf(pdf_url, save_file_path)