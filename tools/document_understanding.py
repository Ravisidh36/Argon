from markitdown import MarkItDown

md = MarkItDown()

def doc_reader(path):
    output = md.convert(path)
    return output.text_content