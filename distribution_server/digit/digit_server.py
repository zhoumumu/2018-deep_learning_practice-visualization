# coding: utf-8

import tornado.httpserver
import tornado.ioloop
import tornado.options
import os
import cgi
import tornado.web
from tornado.options import define, options
from tornado.escape import json_decode, json_encode
from tornado.concurrent import Future

from digit_model import digit_model_instance

define("port", default=8060, help="run on the given port", type=int)

class DigitService(object):

    def do_something(self, *params):
        data = digit_model_instance.api_function(*params)

        res = dict(
            rtn = 200,
            msg = "成功",
            data = data
        )

        return res
    
    def do_something_async(self, *params):
        res_future = Future()

        data = digit_model_instance.api_function(*params)

        res = dict(
            rtn = 200,
            msg = "成功",
            data = data
        )

        res_future.set_result(res)
        
        return res_future

    def upload_image(self, file_metas):
        res_future = Future()

        file_path = None
        if file_metas:
            os.system('rm -rf /home/shixun3/dyk/data/UPLOAD/imgs/*')
            for meta in file_metas:
                # imgs path: /home/shixun3/dyk/data/UPLOAD/imgs
                upload_path = "/home/shixun3/dyk/data/UPLOAD/imgs"

                filename = meta['filename']
                file_path = os.path.join(upload_path, filename)

                with open(file_path, 'wb') as f:
                    f.write(meta['body'])

        res_future.set_result(file_path)

        return res_future


class SyncHandler(tornado.web.RequestHandler):
    def initialize(self):
        self.digit_service = DigitService()
    
    def get(self):
        res = self.digit_service.do_something()

        self.set_status(res.get("rtn"))
        self.set_header("Content-Type", "application/json")
        self.write(json_encode(res))

        self.finish()

    @tornado.gen.coroutine
    def post(self):

        form = cgi.FieldStorage()
        for key in form.keys():
            file_metas = form[key].value
            image_path = yield self.digit_service.upload_image(file_metas)

            if not image_path:
                res = dict(
                    rtn=500,
                    msg="图片上传失败",
                    data={}
                )
                break
        res = yield self.digit_service.do_something("UPLOAD")
        self.set_status(res.get("rtn"))
        self.set_header("Content-Type", "application/json")
        self.write(json_encode(res))

        self.finish()


class AsyncHandler(tornado.web.RequestHandler):
    def initialize(self):
        self.digit_service = DigitService()

    @tornado.gen.coroutine
    def get(self):
        res = yield self.digit_service.do_something_async()

        self.set_status(res.get("rtn"))
        self.set_header("Content-Type", "application/json")
        self.write(json_encode(res))

        self.finish()


    @tornado.gen.coroutine
    def post(self):

        form = cgi.FieldStorage()
        for key in form.keys():
            file_metas = form[key].value
            if key is "label":
                label_path = yield self.digit_service.upload_image(file_metas)
            image_path = yield self.digit_service.upload_image(file_metas)

            if not image_path:
                res = dict(
                    rtn=500,
                    msg="图片上传失败",
                    data={}
                )
                break
        res = yield self.digit_service.do_something_async("UPLOAD")
        self.set_status(res.get("rtn"))
        self.set_header("Content-Type", "application/json")
        self.write(json_encode(res))

        self.finish()


if __name__ == "__main__":
    tornado.options.parse_command_line()
    app = tornado.web.Application(handlers=[
        (r"/api/async_api", AsyncHandler),
        (r"/api/sync_api", SyncHandler),
    ])
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

# http://localhost:8080/api/async_api
# http://localhost:8080/api/sync_api
