import json
import pandas as pd
import pickle
import urllib
from apiclient.discovery import build
from urllib.error import HTTPError, URLError
#import signal

key = 'AIzaSyBYX1tSqkSflKAJjs_KwTOF9HWFgzy4Gig'
videoId = 't_KdbASIkB8'

def build_service():
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"
    return build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=key)

def get_comments(part='snippet', maxResults=100, textFormat='plainText', order='time', videoId=videoId):
    comments, commentsId, authorurls, authornames, repliesCount, likesCount, viewerRating, dates, vidIds, totalReplyCounts, vidTitles = [], [], [], [], [], [], [], [], [], [], []
    
    service = build_service()
    
    try:
        response = service.commentThreads().list(
            part=part,
            maxResults=maxResults,
            textFormat=textFormat,
            order=order,
            videoId=videoId
        ).execute()

        while response:
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comment_id = item['snippet']['topLevelComment']['id']
                reply_count = item['snippet']['totalReplyCount']
                like_count = item['snippet']['topLevelComment']['snippet']['likeCount']
                authorurl = item['snippet']['topLevelComment']['snippet']['authorChannelUrl']
                authorname = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
                date = item['snippet']['topLevelComment']['snippet']['publishedAt']
                vidId = item['snippet']['topLevelComment']['snippet']['videoId']
                totalReplyCount = item['snippet']['totalReplyCount']
                vidTitle = get_vid_title(vidId)

                comments.append(comment)
                commentsId.append(comment_id)
                repliesCount.append(reply_count)
                likesCount.append(like_count)
                authorurls.append(authorurl)
                authornames.append(authorname)
                dates.append(date)
                vidIds.append(vidId)
                totalReplyCounts.append(totalReplyCount)
                vidTitles.append(vidTitle)

            if 'nextPageToken' in response:
                response = service.commentThreads().list(
                    part=part,
                    maxResults=maxResults,
                    textFormat=textFormat,
                    order=order,
                    videoId=videoId,
                    pageToken=response['nextPageToken']
                ).execute()
            else:
                break

    except (HTTPError, URLError) as e:
        print(f"Error: {e}")
       
    finally:
        # Save data before exiting
        data = {
            'comment': comments,
            'comment_id': commentsId,
            'author_url': authorurls,
            'author_name': authornames,
            'reply_count': repliesCount,
            'like_count': likesCount,
            'date': dates,
            'vidid': vidIds,
            'total_reply_counts': totalReplyCounts,
            'vid_title': vidTitles
        }
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['just_date'] = df['date'].dt.date
        df.to_csv('./friedrice_comments.csv')

def get_vid_title(vidid):
    params = {"format": "json", "url": "https://www.youtube.com/watch?v=%s" % vidid}
    url = "https://www.youtube.com/oembed"
    query_string = urllib.parse.urlencode(params)
    url = url + "?" + query_string

    with urllib.request.urlopen(url) as response:
        response_text = response.read()
        data = json.loads(response_text.decode())
        print(data['title'])
        return data['title']

if __name__ == '__main__':
    get_comments()