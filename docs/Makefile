# Path to the package to document
PACKAGE = ../treeIDW

# Output directory for the documentation
DOC_DIR = .

# Path to the logo file
LOGO = logo.png

# URL of the logo on GitHub
LOGO_URL = https://dorian210.github.io/treeIDW/logo.png

# Output PDF filename
OUTPUT_PDF = doc.pdf

# File to store the HTTP server process ID
SERVER_PID_FILE = server.pid

.PHONY: all clean doc

# Default command: generates both HTML and PDF documentation
all: doc pdf

# Generate documentation using pdoc
doc:
	pdoc -o $(DOC_DIR) -d numpy --logo $(LOGO_URL) --favicon $(LOGO_URL) $(PACKAGE)

# Generate a PDF documentation from HTML
pdf:
	@echo "Starting HTTP server..."
	python3 -m http.server 8000 & echo $$! > $(SERVER_PID_FILE); \
	sleep 2; \
	echo "Searching for HTML pages..."; \
	PAGES=$$(find . -name "*.html" | grep -v "index.html" | sed "s|^\./||" | sed "s|^|http://localhost:8000/|" | tac); \
	echo "Found pages: $$PAGES"; \
	echo "Converting HTML pages to PDF..."; \
	wkhtmltopdf $$PAGES $(OUTPUT_PDF); \
	echo "Generated PDF: $(OUTPUT_PDF)"; \
	SERVER_PID=$$(cat $(SERVER_PID_FILE)); \
	echo "Stopping HTTP server (pid=$$SERVER_PID)"; \
	kill $$SERVER_PID; \
	rm -f $(SERVER_PID_FILE)

# Clean up generated files (excluding Makefile and logo)
clean:
	find $(DOC_DIR) -mindepth 1 ! -name 'Makefile' ! -name $(LOGO) -exec rm -rf {} +
