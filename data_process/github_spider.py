import requests
import csv
import time


# Github API token
TOKEN = 'xxx'

# Github API URL
URL = 'https://api.github.com/search/repositories'

# Query parameters
PARAMS = {
    'q': f'\"spring\" in:readme language:Java stars:>100 path:\"src/test/java\"',
    'sort': 'stars',
    'order': 'desc',
    'per_page': 20,
    'page': 1
}

# CSV file to store the results
CSV_FILE = 'spring-github_projects-100stars-before20240101.csv'

# Headers for the CSV file
CSV_HEADERS = ['Repository Name', 'URL', 'Stars', 'Created At', 'Last Pushed', 'Commit Count', 'Has pom.xml']

def get_total_commits(repo):
    contributors_url = repo['contributors_url']
    headers = {'Authorization': f'token {TOKEN}'}
    response = requests.get(contributors_url, headers=headers)
    contributors = response.json()
    total_commits = sum(contributor['contributions'] for contributor in contributors)
    return total_commits

def get_first_commit_date(repo):
    commit_url = f"{repo['url']}/commits?per_page=1"
    headers = {'Authorization': f'token {TOKEN}'}
    response = requests.get(commit_url, headers=headers)
    if response.status_code == 200 and response.json():
        # Get the last page which contains the first commit
        if 'Link' in response.headers:
            links = response.headers['Link'].split(',')
            for link in links:
                if 'rel="last"' in link:
                    last_page_url = link.split(';')[0].strip(' <>')
                    response = requests.get(last_page_url, headers=headers)
                    if response.status_code == 200 and response.json():
                        return response.json()[-1]['commit']['committer']['date']
        else:
            # If there's only one page of commits, then the first commit is in the current response
            return response.json()[-1]['commit']['committer']['date']
    return None

# Function to get the commit count of a repository
def get_commit_count(repo):
    commit_url = repo['commits_url'].split('{')[0]
    headers = {'Authorization': f'token {TOKEN}'}
    response = requests.get(commit_url, headers=headers)
    return len(response.json())

# Function to check if a repository has a pom.xml file
def has_pom_xml(repo):
    contents_url = repo['contents_url'].split('{')[0]
    headers = {'Authorization': f'token {TOKEN}'}
    response = requests.get(contents_url, headers=headers)
    for file in response.json():
        if file['name'] == 'pom.xml':
            return True
    return False

# Function to write the results to a CSV file
def write_to_csv(repos):
    with open(CSV_FILE, 'a', newline='') as file:
        writer = csv.writer(file)
        for repo in repos:
            writer.writerow([repo['name'], repo['html_url'], repo['stargazers_count'], repo['created_at'], repo['pushed_at'], repo['commit_count'], repo['has_pom_xml']])

# Main function
def main():
    with open(CSV_FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(CSV_HEADERS)
    headers = {'Authorization': f'token {TOKEN}'}
    page = 1
    while True:
        try :
            PARAMS['page'] = page
            s = requests.session
            s.keep_alive = False
            response = requests.get(URL, headers=headers, params=PARAMS, verify=False)
            print(response)
            result = response.json()
            if 'items' not in result or not result['items']:
                break
            print("processing page: ", page)
            repos = result['items']
            valid_repos = []
            for repo in repos:
                commit_count = get_total_commits(repo)
                first_commit_date = get_first_commit_date(repo).split('T')[0]
                repo["created_at"] = first_commit_date
                repo['commit_count'] = commit_count
                repo['has_pom_xml'] = has_pom_xml(repo)
                valid_repos.append(repo)
            write_to_csv(valid_repos)
            print("writing to file...")
            page += 1
            time.sleep(10)  # To prevent hitting rate limit
        except Exception as e:
            print(e)
            time.sleep(30)  # To prevent hitting rate limit

if __name__ == '__main__':
    print(PARAMS)
    main()
