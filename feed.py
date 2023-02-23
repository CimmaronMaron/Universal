# feed.py

from functools import cache

import feedparser
import html2text

import reader

@cache
def _get_feed(url=reader.URL):
    """Read the web feed, use caching to only read it once"""
    return feedparser.parse(url)
