from langchain.tools import tool
import os
import requests
import json
import re





@tool
def write_article_to_markdown(article_content: str) -> str:
    """
    Writes an article to a markdown file.

    Parameters:
    - article_content: The content of the article.

    Returns:
    - A string indicating the path of the markdown file.
    """
    title = article_content.split('.')[0].strip() or "Untitled Article"
    tags = re.findall(r'\b[A-Za-z]{4,}\b', article_content)
    tags = list(set(tags))[:5]  # Select unique tags, limit to 5

    filename = f"{title.lower().replace(' ', '_')}.md"
    filename = "article.md"

    frontmatter = f"---\ntitle: {title}\ntags: {tags}\n---\n\n"
    
    with open(filename, 'w') as file:
        file.write(frontmatter + article_content)
    
    return f"Article written to {filename}"

@tool
def publish_to_devto_from_file(filename: str) -> str:
    """
    Publish an article to dev.to from a markdown file.

    Parameters:
    - filename: The markdown file containing the article with frontmatter.

    Returns:
    - A string indicating success or failure of the publishing process.
    """
    api_key = os.getenv('DEVTO_API_KEY')
    if not api_key:
        return "API key not found in environment variables."

    try:
        with open(filename, 'r') as file:
            content = file.read()

        parts = content.split('---\n')
        frontmatter = parts[1].split('\n')
        title = frontmatter[1].split(': ')[1]
        tags = re.findall(r'\[([^\]]+)\]', frontmatter[2])[0].split(', ')
        body_markdown = parts[2]

        headers = {
            'api-key': api_key,
            'Content-Type': 'application/json'
        }
        payload = {
            'article': {
                'title': title,
                'published': False,
                'body_markdown': body_markdown,
                'tags': tags
            }
        }
        response = requests.post('https://dev.to/api/articles', headers=headers, data=json.dumps(payload))

        if response.status_code == 201:
            return 'Article published successfully.'
        else:
            return f'Failed to publish article. Status code: {response.status_code}, Response: {response.text}'
    except Exception as e:
        return f"Failed to publish article. Error: {str(e)}"

@tool
def read_outline_from_file(filename: str) -> str:
    """
    Reads an outline from a markdown file.

    Parameters:
    - filename: The name of the markdown file to read from.

    Returns:
    - A string containing the content of the outline.
    """
    try:
        with open(filename, 'r') as file:
            outline_content = file.read()
        return outline_content
    except FileNotFoundError:
        return "Outline file not found."
    
@tool
def write_outline_to_markdown(outline: str) -> str:
    """
    Writes an outline to a markdown file named 'outline.md'.

    Parameters:
    - outline: The content of the outline in a structured markdown format.

    Returns:
    - A string indicating the path of the markdown file.
    """
    filename = "outline.md"
    with open(filename, 'w') as file:
        file.write(outline)
    return f"Outline written to {filename}"

# Example usage:
# article_content = 'Your article content here...'
# write_article_to_markdown(article_content)
# publish_to_devto_from_file('article.md')
# outline_content = "# Article Title\n\n- Tag1\n- Tag2\n\nRest of the outline..."
# write_outline_to_markdown(outline_content)
