from pathlib import Path
from dataclasses import dataclass, field
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption


@dataclass
class ParsedDocument:
    """Container for a Docling-parsed document."""
    file_path: str
    file_name: str
    markdown_text: str
    docling_document: object        # raw DoclingDocument for HybridChunker
    num_pages: int
    metadata: dict = field(default_factory=dict)


def get_converter() -> DocumentConverter:
    """
    Build a Docling DocumentConverter with optimized settings.
    Table structure detection enabled — critical for research papers.
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False                  # skip OCR — papers are digital
    pipeline_options.do_table_structure = True        # parse tables properly
    pipeline_options.table_structure_options.do_cell_matching = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )


def load_pdf(file_path: str | Path) -> ParsedDocument:
    """
    Load and parse a PDF using Docling.

    Args:
        file_path: Path to the PDF file

    Returns:
        ParsedDocument with markdown text + raw DoclingDocument
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected PDF, got: {path.suffix}")

    print(f"  Parsing '{path.name}' with Docling...")

    converter = get_converter()
    result = converter.convert(str(path))
    doc = result.document

    # Export to markdown for inspection
    markdown_text = doc.export_to_markdown()

    # Page count from Docling
    num_pages = len(doc.pages) if hasattr(doc, 'pages') else 0

    print(f"  Parsed: {num_pages} pages, "
          f"{len(markdown_text):,} chars")

    return ParsedDocument(
        file_path=str(path.absolute()),
        file_name=path.name,
        markdown_text=markdown_text,
        docling_document=doc,
        num_pages=num_pages,
        metadata={
            "source": path.name,
            "file_path": str(path.absolute()),
            "num_pages": num_pages,
            "file_size_kb": round(path.stat().st_size / 1024, 2),
        }
    )


def load_pdfs_from_dir(dir_path: str | Path) -> list[ParsedDocument]:
    """Load all PDFs from a directory."""
    dir_path = Path(dir_path)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    pdf_files = list(dir_path.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDFs found in: {dir_path}")

    print(f"Found {len(pdf_files)} PDF(s)")
    return [load_pdf(f) for f in pdf_files]