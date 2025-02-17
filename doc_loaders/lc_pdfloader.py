from langchain.document_loaders.pdf import PyPDFLoader

# Set the path to your PDF file
pdf_file_path = "TMF_ODAsoftwaremarket_v3.pdf"

# Create a PDFLoader instance
pdf_loader = PyPDFLoader(pdf_file_path)

# Load the documents
document = pdf_loader.load()

# Print the text content of the PDF
print(document[0].page_content)