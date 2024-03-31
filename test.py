from pathlib import Path

# s = "can you tell me more detailed? file_path: 400/440/440.00/440.00.md and else thing?"
# s1 = '.'

# print(Path(s1).is_file())

# def extract_path(q, keyword='file_path: '):    
#     # 'file_path: ' Find path by ketword
    
    
#     start_index = q.find(keyword)
    
#     if keyword not in q:
#         return q

#     if start_index != -1:
#         # 'file_path: ' 
#         start_index += len(keyword)
#         temp_extract = s[start_index:]

#         # Find postion '.md' from temp_extract string
#         end_index = temp_extract.find('.md')

#         if end_index != -1:
#             # Extract final path that include '.md'
#             extracted_path = temp_extract[:end_index + len('.md')]
#             print(extracted_path)
#         else:
#             print("'.md' not found.")
#     else:
#         print("'file_path: ' pattern not found.")

query = "AAA"
file_session = "XYZ123"  # This should be your session identifier or relevant information
file_info = f"Information: {file_session}\n"  # Using an f-string for dynamic insertion

query = file_info + query  # Combining the strings
print(query)