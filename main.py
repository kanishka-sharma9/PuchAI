from typing import Annotated
from fastmcp import FastMCP,server
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
import markdownify
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, TextContent
from pydantic import BaseModel, AnyUrl, Field  # Fixed import
import readabilipy
from pathlib import Path
from playwright.sync_api import Playwright, sync_playwright
from playwright.async_api import async_playwright
from browserbase import Browserbase
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv("./.env")

os.environ["BROWSERBASE_API_KEY"]=os.getenv("BROWSERBASE_API_KEY")
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

bb = Browserbase()

TOKEN = "1e3aa21879b4"
MY_NUMBER = "919811254743"  # Insert your number {91}{Your number}


class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None


class SimpleBearerAuthProvider(BearerAuthProvider):
    """
    A simple BearerAuthProvider that does not require any specific configuration.
    It allows any valid bearer token to access the MCP server.
    For a more complete implementation that can authenticate dynamically generated tokens,
    please use `BearerAuthProvider` with your public key or JWKS URI.
    """

    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(
            public_key=k.public_key, jwks_uri=None, issuer=None, audience=None
        )
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="unknown",
                scopes=[],
                expires_at=None,  # No expiration for simplicity
            )
        return None


class Fetch:
    IGNORE_ROBOTS_TXT = True
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
    ) -> tuple[str, str]:
        """
        Fetch the URL and return the content in a form ready for the LLM, as well as a prefix string with status information.
        """
        from httpx import AsyncClient, HTTPError

        async with AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=30,
                )
            except HTTPError as e:
                raise McpError(
                    ErrorData(
                        code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"
                    )
                )
            if response.status_code >= 400:
                raise McpError(
                    ErrorData(
                        code=INTERNAL_ERROR,
                        message=f"Failed to fetch {url} - status code {response.status_code}",
                    )
                )

            page_raw = response.text

        content_type = response.headers.get("content-type", "")
        is_page_html = (
            "<html" in page_raw[:100] or "text/html" in content_type or not content_type
        )

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (
            page_raw,
            f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
        )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        """Extract and convert HTML content to Markdown format.

        Args:
            html: Raw HTML content to process

        Returns:
            Simplified markdown version of the content
        """
        ret = readabilipy.simple_json.simple_json_from_html_string(
            html, use_readability=True
        )
        if not ret["content"]:
            return "<error>Page failed to be simplified from HTML</error>"
        content = markdownify.markdownify(
            ret["content"],
            heading_style=markdownify.ATX,
        )
        return content

mcp = FastMCP(
    "My MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

ResumeToolDescription = RichToolDescription(
    description="Serve your resume in plain markdown.",
    use_when="Puch (or anyone) asks for your resume; this must return raw markdown, no extra formatting.",
    side_effects=None,
)
NotionSummarizerToolDescription=RichToolDescription(
    description="Return summary of notion page's content.",
    use_when="Puch (or anyone) asks to summarize a notion page",
    side_effects=None
)

from pydantic import BaseModel
# from mcp import tool as mcp_tool
from playwright.async_api import async_playwright
import os
from browserbase import Browserbase # ✅ Correct import

class NotionSummarizerInput(BaseModel):
    url: str

@mcp.tool(description=NotionSummarizerToolDescription.model_dump_json())
async def notion_summarizer(input: NotionSummarizerInput) -> str:
    # ✅ Create remote Browserbase session
    session = Browserbase().sessions.create(project_id=os.getenv("BROWSERBASE_PROJECT_ID"))

    async with async_playwright() as p:
        chromium = p.chromium
        browser = await chromium.connect_over_cdp(session.connect_url)
        context = browser.contexts[0]
        page = context.pages[0]

        try:
            await page.goto(input.url)
            content = await page.text_content('*')
            return await summarize(content)
        finally:
            await page.close()
            await browser.close()

# ✅ Async-compatible summarize using Groq
async def summarize(text: str) -> str:
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="llama-3.1-8b-instant")
    resp = await llm.ainvoke(
        f"Summarize the following Notion page content into structured sections:\n\n{text}"
    )
    return resp.content




# @mcp.tool(description=ResumeToolDescription.model_dump_json())
# async def resume(filename: str = "Kanishka_sharma_Resume (2).pdf") -> str:
#     """
#     Find, read, and convert resume to markdown format.
#     Returns markdown string or error message.
#     """
#     import os
#     import re
#     from pathlib import Path
    
#     try:
#         # Find file
#         if filename:
#             if not os.path.exists(filename):
#                 return f"# Error\nFile '{filename}' not found."
#             target_file = filename
#         else:
#             # Search for common resume names
#             common_names = ['resume.pdf', 'resume.docx', 'resume.txt', 'CV.pdf', 'cv.pdf', 'resume.md']
#             target_file = None
#             for name in common_names:
#                 if os.path.exists(name):
#                     target_file = name
#                     break
#             if not target_file:
#                 return "# Error\nNo resume file found. Upload resume.pdf, resume.docx, or resume.txt"
        
#         # Get file extension
#         ext = Path(target_file).suffix.lower()
        
#         # Read file based on type
#         if ext in ['.txt', '.md']:
#             with open(target_file, 'r', encoding='utf-8') as f:
#                 text = f.read()
#             if ext == '.md':
#                 return text
        
#         elif ext == '.pdf':
#             try:
#                 import PyPDF2
#                 with open(target_file, 'rb') as f:
#                     reader = PyPDF2.PdfReader(f)
#                     text = ""
#                     for page in reader.pages:
#                         text += page.extract_text() + "\n"
#             except ImportError:
#                 return "# Error\nInstall PyPDF2: pip install PyPDF2"
#             except Exception as e:
#                 return f"# Error\nFailed to read PDF: {str(e)}"
        
#         elif ext == '.docx':
#             try:
#                 import docx
#                 doc = docx.Document(target_file)
#                 text = ""
#                 for paragraph in doc.paragraphs:
#                     text += paragraph.text + "\n"
#             except ImportError:
#                 return "# Error\nInstall python-docx: pip install python-docx"
#             except Exception as e:
#                 return f"# Error\nFailed to read DOCX: {str(e)}"
        
#         else:
#             return f"# Error\nUnsupported file type: {ext}"
        
#         # Convert to markdown
#         markdown = text.strip()
        
#         # Basic formatting improvements
#         lines = markdown.split('\n')
#         formatted_lines = []
        
#         for line in lines:
#             line = line.strip()
#             if not line:
#                 formatted_lines.append('')
#                 continue
                
#             # Convert all caps sections to headers
#             if line.isupper() and len(line) > 2:
#                 formatted_lines.append(f'# {line.title()}')
#             # Convert lines ending with colon to subheaders
#             elif line.endswith(':') and len(line) > 3:
#                 formatted_lines.append(f'## {line[:-1]}')
#             else:
#                 # Format email addresses
#                 line = re.sub(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', r'[\1](mailto:\1)', line)
#                 # Format phone numbers
#                 line = re.sub(r'(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})', r'**\1**', line)
#                 # Format URLs
#                 line = re.sub(r'(https?://[^\s]+)', r'[\1](\1)', line)
#                 formatted_lines.append(line)
        
#         markdown = '\n'.join(formatted_lines)
        
#         if not markdown.startswith('#'):
#             markdown = "# Resume\n\n" + markdown
            
#         return markdown
        
#     except Exception as e:
#         return f"# Error\n{str(e)}"


@mcp.tool
async def validate() -> str:
    """
    NOTE: This tool must be present in an MCP server used by puch.
    """
    return MY_NUMBER


FetchToolDescription = RichToolDescription(
    description="Fetch a URL and return its content.",
    use_when="Use this tool when the user provides a URL and asks for its content, or when the user wants to fetch a webpage.",
    side_effects="The user will receive the content of the requested URL in a simplified format, or raw HTML if requested.",
)


@mcp.tool(description=FetchToolDescription.model_dump_json())
async def fetch(
    url: Annotated[AnyUrl, Field(description="URL to fetch")],
    max_length: Annotated[
        int,
        Field(
            default=5000,
            description="Maximum number of characters to return.",
            gt=0,
            lt=1000000,
        ),
    ] = 5000,
    start_index: Annotated[
        int,
        Field(
            default=0,
            description="On return output starting at this character index, useful if a previous fetch was truncated and more context is required.",
            ge=0,
        ),
    ] = 0,
    raw: Annotated[
        bool,
        Field(
            default=False,
            description="Get the actual HTML content if the requested page, without simplification.",
        ),
    ] = False,
) -> list[TextContent]:
    """Fetch a URL and return its content."""
    url_str = str(url).strip()
    if not url_str:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="URL is required"))

    try:
        content, prefix = await Fetch.fetch_url(url_str, Fetch.USER_AGENT, force_raw=raw)
        original_length = len(content)
        
        if start_index >= original_length:
            content = "<error>No more content available.</error>"
        else:
            truncated_content = content[start_index : start_index + max_length]
            if not truncated_content:
                content = "<error>No more content available.</error>"
            else:
                content = truncated_content
                actual_content_length = len(truncated_content)
                remaining_content = original_length - (start_index + actual_content_length)
                # Only add the prompt to continue fetching if there is still remaining content
                if actual_content_length == max_length and remaining_content > 0:
                    next_start = start_index + actual_content_length
                    content += f"\n\n<error>Content truncated. Call the fetch tool with a start_index of {next_start} to get more content.</error>"
        
        return [TextContent(type="text", text=f"{prefix}Contents of {url_str}:\n{content}")]
    
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch URL: {str(e)}"))


async def main():
    try:
        await mcp.run_async(
            "streamable-http",
            host="0.0.0.0",
            port=8085,
        )
    except Exception as e:
        print(f"Server failed to start: {e}")
        raise
    


if __name__ == "__main__":
    import asyncio
    import sys
    asyncio.run(main())