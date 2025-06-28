from playwright.sync_api import Playwright, sync_playwright
from playwright.async_api import async_playwright
from browserbase import Browserbase
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv("./.env")

os.environ["BROWSERBASE_API_KEY"]=os.getenv("BROWSERBASE_API_KEY")

bb = Browserbase()


def run(playwright: Playwright) -> None:
    # Create a session on Browserbase
    session = bb.sessions.create(project_id=os.getenv("BROWSERBASE_PROJECT_ID"))

    # Connect to the remote session
    chromium = playwright.chromium
    browser = chromium.connect_over_cdp(session.connect_url)
    context = browser.contexts[0]
    page = context.pages[0]

    try:
        # Execute Playwright actions on the remote browser tab
        page.goto("https://www.notion.so/Foodery-f3d154a0015647f9993321d851104080?source=copy_link")
        return summarize(page.text_content('*'))
    finally:
        page.close()
        browser.close()

def summarize(text):
    llm=ChatGroq(model="llama-3.1-8b-instant")
    resp= llm.invoke("Extract the page content (main information) from the following and summarize it. Divide into categories.\n\n"
    "<text>" \
    f"{text}"
    "</text>" \
    "Ignore the HTNL,CSS and JS content completely.")
    return resp.content

if __name__ == "__main__":
    with sync_playwright() as playwright:
        print()
        x=run(playwright)
        print(x)