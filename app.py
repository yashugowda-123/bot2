from flask import Flask, request, render_template
from PyPDF2 import PdfReader

app = Flask(_name_)

@app.route('/', methods=['GET', 'POST'])
def upload_pdf():
    if request.method == 'POST':
        file = request.files['pdf']
        reader = PdfReader(file)
        text = reader.pages[0].extract_text()
        return f"<pre>{text}</pre>"
    return '''
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="pdf" accept=".pdf">
            <input type="submit">
        </form>
    '''

if _name_ == '_main_':
    app.run(debug=True)