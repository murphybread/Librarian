from langchain_text_splitters import HTMLHeaderTextSplitter

url = "https://www.murphybooks.me/projects/library/000/020/020-10/020-10-a/"

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
    ("pre", "Preformatted"),  # Treat 'pre' tags as sections to split on.
    ("code", "Code")          # Optionally, if you want to distinguish code blocks explicitly. 
]

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
print(html_splitter)
html_header_splits = html_splitter.split_text_from_url(url)
print(html_header_splits)

for i in range(1,len(html_header_splits)):
    print(i)
    print(html_header_splits[i].page_content[:500])

# from langchain_text_splitters import HTMLHeaderTextSplitter

# html_string = """
# <!DOCTYPE html>
# <html>
# <body>
#     <div>
#         <h1>Foo</h1>
#         <p>Some intro text about Foo.</p>
#         <div>
#             <h2>Bar main section</h2>
#             <p>Some intro text about Bar.</p>
#             <h3>Bar subsection 1</h3>
#             <p>Some text about the first subtopic of Bar.</p>
#             <h3>Bar subsection 2</h3>
#             <p>Some text about the second subtopic of Bar.</p>
#         </div>
#         <div>
#             <h2>Baz</h2>
#             <p>Some text about Baz</p>
#         </div>
#         <br>
#         <p>Some concluding text about Foo</p>
#     </div>
# </body>
# </html>
# """

# headers_to_split_on = [
#     ("h1", "Header 1"),
#     ("h2", "Header 2"),
#     ("h3", "Header 3"),
# ]

# html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
# print(html_splitter)
# html_header_splits = html_splitter.split_text(html_string)

# print(html_header_splits)
