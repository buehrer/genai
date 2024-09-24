# This program is a server that shows NFL scores

from flask import Flask, jsonify
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the NFL Scores Server!"

@app.route('/scores')
def scores():
    # Scrape the web to get the scores
    url = "https://www.espn.com/nfl/scoreboard"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Parse the HTML to get the scores
    score_containers = soup.find_all('div', class_='ScoreboardScoreCell__TeamName')
    scores = []
    
    for i in range(0, len(score_containers), 2): 
        if i + 1 < len(score_containers):
            team1 = score_containers[i].text
            team2 = score_containers[i+1].text
            score = f"{team1} vs {team2}"
            scores.append(score)
    
    # Return the scores
    return jsonify({"scores": scores})

if __name__ == '__main__':
    app.run(debug=True) 