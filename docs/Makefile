# Path to the package to document
PACKAGE = ../treeIDW

# Output directory for the documentation
DOC_DIR = .

# Path to the logo file
LOGO = logo.png

# Output PDF filename
OUTPUT_PDF = doc.pdf

# File to store the HTTP server process ID
SERVER_PID_FILE = server.pid

.PHONY: all clean html pdf

# Default command: generates both HTML and PDF documentation
all: html pdf

# Generate documentation using pdoc and correct the logo path
html:
	pdoc -o $(DOC_DIR) -d numpy --logo $(LOGO) --favicon $(LOGO) $(PACKAGE)
	find . -type f -name "*.html" | while read file; do \
	    depth=$$(dirname "$$file" | awk -F'/' '{print NF}'); \
	    if [ $$depth -eq 1 ]; then \
	        dir_name=$${PWD##*/}; \
	        sed -i "s|$$dir_name/$(LOGO)|$(LOGO)|g" "$$file"; \
	        sed -i "s|$(LOGO)|$(LOGO)|g" "$$file"; \
	    else \
	        rel_path=$$(printf '../%.0s' $$(seq $$((depth - 1))))$(LOGO); \
	        sed -i "s|$(LOGO)|$$rel_path|g" "$$file"; \
	    fi; \
	done

# Generate a PDF documentation from HTML
pdf:
	@echo "Starting HTTP server..."
	python3 -m http.server 8000 & echo $$! > $(SERVER_PID_FILE); \
	sleep 2; \
	echo "Searching for HTML pages..."; \
	PAGES=$$(find . -type d -print | sort | while read dir; do \
		find "$$dir" -maxdepth 1 -name "*.html" -not -name "index.html" | \
		sed "s|^\./|http://localhost:8000/|"; done); \
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
