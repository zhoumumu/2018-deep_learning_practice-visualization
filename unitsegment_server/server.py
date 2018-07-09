# coding: utf-8
# 用了 Python 的 Tornado 框架，当然也可以使用其它框架
# 英文文档 (v5.0) http://www.tornadoweb.org/en/stable/
# 中文文档 (v4.3) http://tornado-zh.readthedocs.io/zh/latest/
import os

import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from tornado.options import define, options
from tornado.escape import json_decode, json_encode
from tornado.concurrent import Future

# 这里导入模型实例
from model import cnnvisualizer_instance as cvi

# 定义端口默认值
define("port", default=8085, help="run on the given port", type=int)

# 处理请求的 Handler，匹配正则表达式 api
class SegmentHandler(tornado.web.RequestHandler):
    def initialize(self):
        # 初始化时引入对应的 Service
        self.segment_service = SegmentService()

    # tornado 框架的协程装饰器，通过协程实现异步。
    # 也可以直接写同步代码：去掉@tornado.gen.coroutine，Service 也不用返回 Future 对象。
    @tornado.gen.coroutine
    def post(self):
        # [Request] (multipart/form-data)
        # {
        #     "name": "img",
        #     "file": "xxx.jpg",
        # }

        file_metas = self.request.files.get("img")
        layer_metas = self.get_body_argument("layer")
        unit_metas = self.get_body_argument("unit")
        unit_metas = int(unit_metas) 

        print(layer_metas, type(layer_metas))
        print(unit_metas, type(unit_metas))

        # 上传图片，返回图片路径
        # yield Future 对象，将异步转化为同步写法
        image_path = yield self.segment_service.upload_image(file_metas)
        print(image_path) 

        res = dict(
            rtn = 200,
            msg = "test"
        )

        if image_path:
            res = yield self.segment_service.segment_model_run(
                image_path, layer_metas, unit_metas)
        else:
            res = dict(
                rtn = 500,
                msg = "upload image error",
                data = {}
            )

        # 写 http 状态码，header，body
        self.set_status(res.get("rtn"))
        self.set_header("Content-Type", "application/json")
        self.write(json_encode(res))

        self.finish()

# Service，每个组按自己的写就好
class SegmentService(object):
    def upload_image(self, file_metas):
        # 创建 Future 对象，若采用同步写法则不需要
        # Future 对象可以简单理解为类似 js promise 的东西
        res_future = Future()

        file_path = None
        if (file_metas):
            for meta in file_metas:
                # 上传图片保存路径为 {当前目录}/realimg
                upload_path = os.path.join(os.path.dirname(__file__), "static/inputs")
                
                filename = meta['filename']
                file_path = os.path.join(upload_path, filename)

                with open(file_path, 'wb') as f:
                    f.write(meta['body'])
        
        # 将结果 set_result 到 Future 对象中即可被 yield
        res_future.set_result(file_path)
        
        # 返回 Future 对象
        return res_future
    
    def segment_model_run(self, image_path, layer, unit):
        # 创建 Future 对象，若采用同步写法则不需要
        res_future = Future()
        res = dict(
            rtn = 0,
            msg = "",
            data = {}
        )
        # 调用模型 API
        try:
            basename = cvi.generate_unitsegment(image_path, layer, unit)

            res["rtn"] = 200
            res["msg"] = "success"
            res["data"] = dict(
                path = os.path.join('http://222.200.180.105:8085/static/outputs/', basename)
            )

        except Exception as e:
            res["rtn"] = 500
            res["msg"] = str(e)
        
        # 将结果 set_result 到 Future 对象中
        res_future.set_result(res)

        # 返回 Future 对象
        return res_future


if __name__ == "__main__":
    tornado.options.parse_command_line()
    app = tornado.web.Application(handlers=[
        (r"/api/segment", SegmentHandler),
        (r"/static/outputs/(.*)", tornado.web.StaticFileHandler,
         {"path": "./static/outputs/"})
    ])
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    print(r"server is running on 127.0.0.1:%d" % options.port)
    tornado.ioloop.IOLoop.instance().start()
