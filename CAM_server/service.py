import json
import tornado.httpserver
import tornado.ioloop
import tornado.web
from tornado.options import define, options
from tornado.escape import json_decode, json_encode
from tornado.concurrent import Future
import os
import cnn as de
import fast_gradient_sign_targeted as fe

define("port", default=7070, help="run on the given port", type=int)
define("host", default="127.0.0.1", help="run on the given port", type=str)

class DetectionService(object):
    def upload_image(self, file_metas):
        # 创建 Future 对象，若采用同步写法则不需要
        # Future 对象可以简单理解为类似 js promise 的东西
        res_future = Future()
        #print(file_metas)
        file_path = None
        if (file_metas):
            for meta in file_metas:
                # 上传图片保存路径为 {当前目录}/realimg
                upload_path = os.path.join(
                    os.path.dirname(__file__), "realimg")
                filename = meta['filename']
                file_path = os.path.join(upload_path, filename)
                #print(file_path)
                with open(file_path, 'wb') as f:
                    f.write(meta['body'])
        # 将结果 set_result 到 Future 对象中即可被 yield
        res_future.set_result(file_path)
        #print(res_future)
        # 返回 Future 对象
        return res_future

    def detection_model_run(self, image_path):
        # 创建 Future 对象，若采用同步写法则不需要
        res_future = Future()
        res = dict(
            rtn=0,
            msg="",
            data={}
        )
        # 调用模型 API
        try:
            #print(image_path)
            campath, pre, idx, _ = de.returnpredict(image_path, 1)
            gampath = fe.generate_tar_ad_sample(image_path, idx)
            gamcampath, gampre, _, _ = de.returnpredict(gampath, 1)
            #print(image_path)
            imgpath = image_path.split("/")[-2] + "/" + image_path.split("/")[-1]
            picurl = options.host + ":" + \
                "6060" + "/api/" + imgpath
            campicurl = options.host + ":" + \
                "6060" + "/api/" + campath[2:]
            gampicurl = options.host + ":" + \
                "6060" + "/api/" + gampath[2:]
            gamcampicurl = options.host + ":" + \
                "6060" + "/api/" + gamcampath[2:]
            #print(picurl)
            res["rtn"] = 200
            res["msg"] = "成功检测该图像"
            # res["data"] = dict(
            #     bboxs = bboxs
            # )
            res["data"] = dict(
                campath=campath,
                predict=pre,
                picture=picurl,
                campicture=campicurl,
                gampicture=gampicurl,
                gampredict=gampre,
                gamcampicture=gamcampicurl
            )
            #print(res)

        except Exception as e:
            res["rtn"] = 500
            res["msg"] = str(e)

        # 将结果 set_result 到 Future 对象中
        res_future.set_result(res)

        # 返回 Future 对象
        return res_future


# Service，每个组按自己的写就好
class DetectionUrlService(object):
    def detection_model_run(self, image_path):
        # 创建 Future 对象，若采用同步写法则不需要
        res_future = Future()
        res = dict(
            rtn=0,
            msg="",
            data={}
        )
        # 调用模型 API
        try:
            campath, pre, idx, img_root = de.returnpredict(image_path, 0)
            print(img_root)
            gampath = fe.generate_tar_ad_sample(img_root, idx)
            gamcampath, gampre, _, _ = de.returnpredict(gampath, 1)
            imgpath = img_root.split(
                "/")[-2] + "/" + img_root.split("/")[-1]
            picurl = options.host + ":" + \
                "6060" + "/api/" + imgpath
            campicurl = options.host + ":" + \
                "6060" + "/api/" + campath[2:]
            gampicurl = options.host + ":" + \
                "6060" + "/api/" + gampath[2:]
            gamcampicurl = options.host + ":" + \
                "6060" + "/api/" + gamcampath[2:]
            #print(picurl)
            res["rtn"] = 200
            res["msg"] = "成功检测该图像"
            # res["data"] = dict(
            #     bboxs = bboxs
            # )
            res["data"] = dict(
                campath=campath,
                predict=pre,
                picture=picurl,
                campicture=campicurl,
                gampicture=gampicurl,
                gampredict=gampre,
                gamcampicture=gamcampicurl
            )

        except Exception as e:
            res["rtn"] = 500
            res["msg"] = str(e)
        # 将结果 set_result 到 Future 对象中
        res_future.set_result(res)
        # 返回 Future 对象
        return res_future
