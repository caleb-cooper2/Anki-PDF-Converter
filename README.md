# PDF to Anki Flash Card Converter

A powerful, easy to use web application that converts PDF lecture slides into Anki flashcards using OpenAI's GPT models.

## Setup

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

On Windows:
```cmd
set OPENAI_API_KEY=your-api-key-here
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5001
```

## Usage

1. **Upload PDF**: Drag and drop your PDF file or click to browse
2. **Configure Parameters**:
   - Deck Name: Name for your Anki deck
   - Max Cards: Total number of flashcards to generate
   - Max Cards Per Slide: Cards per slide (default: 3)
   - Model: Choose your OpenAI model (GPT-4o Mini recommended)
   - Page Range: Specify start and end pages if needed
   - Options: Skip empty slides, generate Anki package
3. **Generate**: Click "Generate Flashcards" and wait for processing
4. **Download**: Once complete, download your CSV or Anki deck file

## Configuration

### PDF Page Limit

The application enforces a maximum of 100 pages per PDF. This can be adjusted in `app.py`:

```python
MAX_PDF_PAGES = 100  # Change this value
```

### File Size Limit

Maximum file size is set to 50MB. Adjust in `app.py`:

```python
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
```
