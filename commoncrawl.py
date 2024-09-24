import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

#make a valid filename
def make_filename(url):
    return url.replace(':','').replace('//','').replace('?','').replace('&','').replace('/','_').replace(".","") + '.html'

def simple_web_crawler(start_url):
    visited_urls = set()
    urls_to_visit = [start_url]

    while urls_to_visit:
        current_url = urls_to_visit.pop(0)
        if current_url in visited_urls:
            continue

        print(f"Visiting: {current_url}")
        try:
            response = requests.get(current_url)
            if response.status_code != 200:
                print("response code is " + str(response.status_code))
                continue

            visited_urls.add(current_url)
            #write file to C:/projects/crawler/data
            with open('C:/projects/crawler/' + make_filename(current_url), 'w', encoding="utf-8") as file:
                file.write(response.text)
            soup = BeautifulSoup(response.text, 'html.parser')

            for link in soup.find_all('a', href=True):
                absolute_link = urljoin(current_url, link['href'])
                if absolute_link not in visited_urls and absolute_link.find("www.pro-football-reference.com") != -1:
                    urls_to_visit.append(absolute_link)

        except requests.RequestException:
            continue

if __name__ == "__main__":
    start_url = "http://www.pro-football-reference.com"  # Change this to your starting URL
    simple_web_crawler(start_url)
