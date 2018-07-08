import json
import tornado.httpserver
import tornado.ioloop
import tornado.web
from tornado.escape import json_decode, json_encode
from tornado.concurrent import Future
import service

class DetectionHandler(tornado.web.RequestHandler):
    def initialize(self):
        # 初始化时引入对应的 Service
        self.detection_service = service.DetectionService()
    # tornado 框架的协程装饰器，通过协程实现异步。
    # 也可以直接写同步代码：去掉@tornado.gen.coroutine，Service 也不用返回 Future 对象。
    @tornado.gen.coroutine
    def post(self):
        # [Request] (multipart/form-data)
        # {
        #     "name": "img",
        #     "file": "xxx.jpg"
        # }
        file_metas = self.request.files.get("img")
        #print(file_metas)
        # 上传图片，返回图片路径
        # yield Future 对象，将异步转化为同步写法
        image_path = yield self.detection_service.upload_image(file_metas)
        #print(image_path)
        if image_path:
            res = yield self.detection_service.detection_model_run(image_path)
        else:
            res = dict(
                rtn=500,
                msg="图片上传失败",
                data={}
            )
        # 写 http 状态码，header，body
        self.set_status(res.get("rtn"))
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(res, ensure_ascii=False).replace("</", "<\\/"))
        self.finish()


class DetectionUrlHandler(tornado.web.RequestHandler):
    def initialize(self):
        # 初始化时引入对应的 Service
        self.detection_service = service.DetectionUrlService()
    # tornado 框架的协程装饰器，通过协程实现异步。
    # 也可以直接写同步代码：去掉@tornado.gen.coroutine，Service 也不用返回 Future 对象。
    @tornado.gen.coroutine
    def post(self):
        # [Request]
        # {
        #     "imgurl": ""
        # }
        #file_metas = self.request.files.get("img")
        data = json.loads(self.request.body)
        imgurl = data['imgurl']
        if imgurl:
            res = yield self.detection_service.detection_model_run(imgurl)
        else:
            res = dict(
                rtn=500,
                msg="图片获取失败",
                data={}
            )
        # 写 http 状态码，header，body
        self.set_status(res.get("rtn"))
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(res, ensure_ascii=False).replace("</", "<\\/"))
        self.finish()
