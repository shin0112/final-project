from langchain.document_loaders import WebBaseLoader

url = {
    "EU": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=COM%3A2023%3A0166%3AFIN",
    "FTC": "https://www.ftc.gov/sites/default/files/attachments/press-releases/ftc-issues-revised-green-guides/greenguides.pdf"
}

# EU loader
loader_EU = WebBaseLoader(url["EU"])
docs_EU = loader_EU.load()

# FTC loader
loader_FTC = WebBaseLoader(url["FTC"])
docs_FTC = loader_FTC.load()
