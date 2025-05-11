import nltk
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import plotext as plt

nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')
nltk.download('maxent_ne_chunker_tab')

async def fetch_bluesky(query: str, limit: int = 10):
    url = f'https://bsky.app/search?q={query}'
    posts = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        print(f"Fetching content: {url}")
        
        try:
            await page.goto(url, timeout=60000)
            print(f"Loaded content: {url}")

            
            await page.wait_for_selector('div.css-146c3p1[data-testid="postText"]', timeout=10000)
            content_html = await page.content()
            soup = BeautifulSoup(content_html, 'html.parser')
            
            tweets = soup.find_all('div', class_='css-146c3p1', attrs={'data-testid': 'postText'}, limit=limit)
            if tweets:
                for tweet in tweets:
                    content = tweet.text.strip()
                    posts.append({'Content': content})
                    
        except Exception as e:
            print(e)
        finally:
            await page.close()

    df = pd.DataFrame(posts)
    return df

def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]

    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

async def main():
    query = "Nick Robertson"
    limit = 25
    df = await fetch_bluesky(query, limit)  

    def calculate_nltk_scores(text):
        sentiment = sia.polarity_scores(text)
        return sentiment

    result = {}

    for i, row in df.iterrows():
        try:
            text = row['Content']
            nltk_result = calculate_nltk_scores(text)

            result[i] = nltk_result
        except RuntimeError:
            continue

    results_df = pd.DataFrame(result).T
    results_df = results_df.merge(df, left_index=True, right_index=True)
    print(results_df)
    if results_df.empty:
        print("No results found")
        return

    plt.title(f"Sentiment Analysis of {query} on BlueSky")
    plt.xlabel("Positive Sentiment")
    plt.ylabel("Negative Sentiment")
    plt.plot(results_df['pos'], results_df['neg'])
    plt.show()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
    