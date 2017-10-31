import base64

from icrawler import ImageDownloader
from icrawler.builtin import GoogleImageCrawler
from six.moves.urllib.parse import urlparse
import urllib.request
import re
import requests

class PrefixNameDownloader(ImageDownloader):
    def get_filename(self, task, default_ext):

        url_path = urlparse(task['file_url'])[2]
        if '.' in url_path:
            extension = url_path.split('.')[-1]
            if extension.lower() not in [
                'jpg', 'jpeg'
            ]:
                extension = default_ext
        else:
            extension = default_ext

        filename = super(PrefixNameDownloader, self).get_filename(
            task, default_ext)
        return 'prefix_' + filename


class Base64NameDownloader(ImageDownloader):
    def get_filename(self, task, default_ext):
        url_path = urlparse(task['file_url'])[2]


        if '.' in url_path:
            extension = url_path.split('.')[-1]
            if extension.lower() not in [
                'jpg', 'jpeg'
            ]:
                extension = default_ext
        else:
            extension = default_ext
        # works for python 3
        filename = base64.b64encode(url_path.encode()).decode()
        return '{}.{}'.format(filename, extension)


# 1.  your_image_dir/받을 object명 입력

google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=4, downloader_cls=PrefixNameDownloader,
                                    storage={'root_dir': 'your_image_dir/cat2'})





# 2. Keyword에 object명 입력 후 실행

google_crawler.crawl(keyword='고양이', max_num=1000,
                     date_min=None, date_max=None,
                     min_size=(200,200), max_size=None)