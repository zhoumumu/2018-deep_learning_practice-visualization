# coding: utf-8
# 用了 Python 的 Tornado 框架，当然也可以使用其它框架
# 英文文档 (v5.0) http://www.tornadoweb.org/en/stable/
# 中文文档 (v4.3) http://tornado-zh.readthedocs.io/zh/latest/
import os

import json
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from tornado.options import define, options
from tornado.escape import json_decode, json_encode
from tornado.concurrent import Future

# 定义端口默认值

#define("post", default=7070, help="run on the given port", type=int)
port = 6060

if __name__ == "__main__":
    #print(options.post)
    tornado.options.parse_command_line()
    app = tornado.web.Application(handlers=[
        (r"/api/realimg/(.*)", tornado.web.StaticFileHandler,
         {"path": "./realimg/"}),
        (r"/api/CAM_output/(.*)", tornado.web.StaticFileHandler,
         {"path": "./CAM_output/"}),
        (r"/api/generated/(.*)", tornado.web.StaticFileHandler,
         {"path": "./generated/"})
    ])
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(port)
    tornado.ioloop.IOLoop.instance().start()
