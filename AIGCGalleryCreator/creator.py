import base64
import os
import re
import threading
import time
from datetime import datetime
from enum import Enum
from os.path import exists

import cv2
import numpy as np
from PIL import Image, ImageDraw
from flask import Flask, Blueprint, request, jsonify
import ollama

from manageQueue import ManageQueue
from taskTrace import TaskStatus, Executor, ImageTCB, VideoTCB

app = Flask(__name__)

blue_painter = Blueprint("blue-painter", __name__)

# 作品生成等待队列
waitingQueue = ManageQueue()
# 作品生成完成队列
finishedQueue = ManageQueue()
# 作品生成异常队列
exceptionQueue = ManageQueue()

# noinspection PyTypeChecker
executor: Executor = None
protect_thread = None

# 可重入锁保障线程安全
executorLock = threading.RLock()


def protectThread():
    """
    守护线程
    """
    global executor
    while True:
        # 清理作品生成异常队列
        exceptionQueue.clear(10)
        with executorLock:
            if executor is not None:
                checkTime = 30 if isinstance(executor.tcb, VideoTCB) else 16
                # 判断当前生成任务是否处于活跃状态
                if time.time() - executor.queryTime >= checkTime:
                    # 当前生成任务已有五秒时间未被激活
                    executor.stop()
                # 若当前存在被中断的生成任务或已完成的生成任务未进入下一阶段或出现异常的生成任务
                generateStatus = executor.generateStatus
                if generateStatus['finish']:
                    # 完成生成任务运行阶段
                    print('The protect thread leaves the creator empty.')
                    executor = None
            elif executor is None and not waitingQueue.isEmpty():
                # 若当前没有正在运行生成任务且等待队列不为空
                tcb = waitingQueue.peek()
                # 输出即将执行的生成任务的信息
                print('\n' + '-' * 10)
                print(f"Run task: < {tcb.taskId} >, Prompt: {tcb.prompt}")
                print('-' * 10 + '\n')
                # 创建生成任务输出保存文件夹
                if not exists(f'./transfer/{tcb.taskId}'):
                    os.mkdir(f'./transfer/{tcb.taskId}')
                # 设置当前的生成任务为运行态
                tcb.taskStatus = TaskStatus.RUNNING
                executor = Executor(tcb, finishedQueue, exceptionQueue, time.time())
                # 将当前运行的生成任务的控制块从作品生成等待队列中移除
                waitingQueue.pop()
                creatorThread = threading.Thread(target=executor.run, name="creatorThread")
                creatorThread.start()
        time.sleep(2)


class QueryStatus(Enum):
    """
    生成任务查询状态
    """
    NO_TASK = 0  # 生成任务不存在
    WAITING_TASK = 1  # 生成任务等待中
    RUNNING_TASK = 2  # 生成任务运行中
    FINISH_TASK = 3  # 生成任务已完成
    ABNORMAL_TASK = 4  # 生成任务出现异常


def getTaskInfo(taskId):
    """
    获取生成任务当前信息
    """
    # 在作品生成完成队列中查找
    for tcb in finishedQueue:
        if tcb.taskId == taskId:
            return QueryStatus.FINISH_TASK, tcb
    # 被查询的生成任务是否为当前运行中的任务
    with executorLock:
        if executor is not None and taskId == executor.tcb.taskId:
            return QueryStatus.RUNNING_TASK, executor.tcb
    # 在作品生成等待队列中查找
    for tcb in waitingQueue:
        if tcb.taskId == taskId:
            return QueryStatus.WAITING_TASK, tcb
    # 在作品生成异常队列中查找
    for tcb in exceptionQueue:
        if tcb.taskId == taskId:
            return QueryStatus.ABNORMAL_TASK, tcb
    # 未发现当前查询的生成任务
    return QueryStatus.NO_TASK, None


# 提交图片或视频生成任务
@blue_painter.route('/creator/submitMediaTask', methods=['POST'])
def submitMediaTask():
    # noinspection PyBroadException
    try:
        global protect_thread
        if protect_thread is None:
            # 首次下达生成任务时启动守护进程
            protect_thread = threading.Thread(target=protectThread, name="protectThread")
            protect_thread.start()
            print('Start protect thread.')
        global executor
        # 打印作品生成等待队列和作品生成完成队列
        print('\n' + '-' * 10)
        print('Waiting: ', end="")
        waitingQueue.show()
        print('Executor: ', end="")
        print(Executor)
        print('Finished: ', end="")
        finishedQueue.show()
        print('-' * 10 + '\n')
        # 读取生成任务请求信息
        taskId = request.form.get('taskId')
        prompt = request.form.get('prompt')
        if re.fullmatch(r'^[A-Za-z0-9 .,!?\-]+$', prompt.replace('\n', '')) is None:
            # 若提示词不符合生成任务要求
            modifyPrompt = "请将以下句子翻译为英文，单词个数控制在45个以内：" + prompt
            prompt = ollama.generate(model='qwen2.5:0.5b', prompt=modifyPrompt)['response']
        createType = request.form.get('createType')
        if createType == 'image':
            modelType = request.form.get('modelType')
            if modelType == 'VQGAN':
                style = request.form.get('style')
                if style == '默认':
                    prompt = [prompt]
                elif style == '艺术':
                    prompt = [prompt, 'ArtStation']
                elif style == '虚幻':
                    prompt = [prompt, 'Unreal Engine']
                elif style == '极简':
                    prompt = [prompt, 'Minimalist']
                else:
                    return jsonify({'success': False, 'result': 'Wrong generate style.'})
            iteration = request.form.get('iteration')
            tcb = ImageTCB(taskId, prompt, modelType, int(iteration))
        elif createType == 'video':
            modelType = request.form.get('modelType')
            iteration = request.form.get('iteration')
            duration = request.form.get('duration')[:-1]
            tcb = VideoTCB(taskId, prompt, modelType, int(iteration), int(duration))
        else:
            return jsonify({'success': False, 'result': 'Wrong media type.'})
        # 将生成任务放入等待队列
        waitingQueue.push(tcb)
        with executorLock:
            if executor is not None:
                # 若当前存在正在执行的生成任务
                return jsonify({'success': True, 'result': 'Waiting.'})
            else:
                # 若当前不存在正在执行的生成任务
                return jsonify({'success': True, 'result': 'Running.'})
    except Exception as e:
        return jsonify({'success': False, 'result': e})


# 查询图片或视频生成任务状态
@blue_painter.route('/creator/enquireMediaTask', methods=['POST'])
def enquireMediaTask():
    try:
        global executor
        taskId = request.form.get('taskId')
        status, tcb = getTaskInfo(taskId)
        if status == QueryStatus.FINISH_TASK:
            # 生成任务已完成
            if isinstance(tcb, ImageTCB):
                # 若生成任务为图像生成任务
                taskSavePath = f'./transfer/{taskId}'
                if not exists(taskSavePath):
                    # 已生成作品丢失
                    finishedQueue.remove(tcb)
                    return jsonify({'success': False, 'result': 'The work has been lost.'})
                # 返回临时存放的最终作品
                currentTime = datetime.now().timestamp() * 1000
                finalImage = None
                minTimeDiff = float('inf')
                # 寻找最新保存的作品
                for filename in os.listdir(taskSavePath):
                    # 去除文件后缀名仅保留时间戳部分
                    imageSaveTime = float(filename[:-4])
                    timeDiff = abs(currentTime - imageSaveTime)
                    if timeDiff < minTimeDiff:
                        minTimeDiff = timeDiff
                        finalImage = os.path.join(taskSavePath, filename)
                if finalImage is None:
                    # 若作品不存在
                    finishedQueue.remove(tcb)
                    return jsonify({'success': False, 'result': 'The work has been lost.'})
                image = cv2.imread(finalImage)
                if image is None:
                    # 若作品无法读取
                    finishedQueue.remove(tcb)
                    return jsonify({'success': False, 'result': 'The image cannot be read.'})
                # 获得图像长宽
                imageHeight, imageWidth = image.shape[:2]
                # 将图像转换为base64编码
                _, imageData = cv2.imencode('.png', image)
                encodedImageData = f"data:image/png;base64,{base64.b64encode(imageData.tobytes()).decode('utf-8')}"
                responseData = {
                    'tempId': taskId,
                    'image': encodedImageData,
                    'mediaWidth': imageWidth,
                    'mediaHeight': imageHeight,
                    'aspectRatio': str(round(imageWidth / imageHeight, 3)),
                    'type': 'image',
                    'modelType': tcb.modelType,
                    'iteration': tcb.iteration
                }
                finishedQueue.remove(tcb)
                # 返回生成任务结果
                return jsonify({'success': True, 'result': {'status': 'complete', 'work': responseData}})
            elif isinstance(tcb, VideoTCB):
                # 若生成任务为视频生成任务
                taskSavePath = f'./transfer/{taskId}'
                if not exists(taskSavePath):
                    # 已生成作品丢失
                    finishedQueue.remove(tcb)
                    return jsonify({'success': False, 'result': 'The work has been lost.'})
                videoList = os.listdir(taskSavePath)
                if len(videoList) == 0:
                    # 若作品不存在
                    finishedQueue.remove(tcb)
                    return jsonify({'success': False, 'result': 'The work has been lost.'})
                finalVideo = os.path.join(taskSavePath, videoList[0])
                video = cv2.VideoCapture(finalVideo)
                if not video.isOpened():
                    # 若作品无法读取
                    finishedQueue.remove(tcb)
                    return jsonify({'success': False, 'result': 'The video cannot be read.'})
                # 获取视频封面
                ret, cover = video.read()
                video.release()
                if not ret:
                    # 若作品封面无法读取
                    finishedQueue.remove(tcb)
                    return jsonify({'success': False, 'result': 'Video cover cannot be obtained.'})
                # 获取视频以及封面的宽高
                videoHeight, videoWidth = cover.shape[:2]
                # 在封面上绘制视频播放提示按钮
                cover = Image.fromarray(cv2.cvtColor(cover, cv2.COLOR_BGR2RGB))
                mask = Image.new("RGBA", cover.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(mask)
                playWidth = int(videoWidth * 0.08)
                playHeight = int(playWidth * 1.155)
                playPoints = [
                    (0.05 * videoWidth, 0.05 * videoHeight),
                    (0.05 * videoWidth + playWidth, 0.05 * videoHeight + playHeight // 2),
                    (0.05 * videoWidth, 0.05 * videoHeight + playHeight)
                ]
                draw.polygon(playPoints, fill=(255, 255, 255, 255))
                cover = Image.alpha_composite(cover.convert("RGBA"), mask)
                cover = cv2.cvtColor(np.array(cover), cv2.COLOR_RGBA2BGR)
                # 将封面转换为base64编码
                _, coverData = cv2.imencode('.png', cover)
                encodedCoverData = f"data:image/png;base64,{base64.b64encode(coverData.tobytes()).decode('utf-8')}"
                # 将视频转换为base64编码
                with (open(finalVideo, "rb") as f):
                    videoData = f.read()
                    encodedVideoData = f"data:video/mp4;base64,{base64.b64encode(videoData).decode('utf-8')}"
                responseData = {
                    'tempId': taskId,
                    'video': encodedVideoData,
                    'cover': encodedCoverData,
                    'mediaWidth': videoWidth,
                    'mediaHeight': videoHeight,
                    'aspectRatio': str(round(videoWidth / videoHeight, 3)),
                    'type': 'video',
                    'modelType': tcb.modelType,
                    'iteration': tcb.duration
                }
                finishedQueue.remove(tcb)
                # 返回生成任务结果
                return jsonify({'success': True, 'result': {'status': 'complete', 'work': responseData}})
            else:
                finishedQueue.remove(tcb)
                return jsonify({'success': False, 'result': 'Unknown task type.'})
        elif status == QueryStatus.RUNNING_TASK:
            # 生成任务正在运行
            if isinstance(tcb, ImageTCB):
                with executorLock:
                    progress = str(int(executor.getCurIter() / tcb.iteration * 100))
                    executor.queryTime = time.time()
                taskSavePath = f'./transfer/{taskId}'
                if not exists(taskSavePath):
                    # 已生成作品丢失
                    return jsonify({'success': False, 'result': 'The work has been lost.'})
                # 返回临时存放的生成中作品
                currentTime = datetime.now().timestamp() * 1000
                finalImage = None
                minTimeDiff = float('inf')
                # 寻找最新保存的作品
                for filename in os.listdir(taskSavePath):
                    # 去除文件后缀名仅保留时间戳部分
                    imageSaveTime = float(filename[:-4])
                    timeDiff = abs(currentTime - imageSaveTime)
                    if timeDiff < minTimeDiff:
                        minTimeDiff = timeDiff
                        finalImage = os.path.join(taskSavePath, filename)
                if finalImage is None or finalImage == tcb.inspectRecord:
                    # 若最新作品不存在或该作品已发送
                    return jsonify({'success': True, 'result': {'status': 'creating', 'progress': progress}})
                image = cv2.imread(finalImage)
                if image is None:
                    # 若最新作品无法读取
                    return jsonify({'success': True, 'result': {'status': 'creating', 'progress': progress}})
                # 将图像转换为base64编码
                _, imageData = cv2.imencode('.png', image)
                encodedImageData = f"data:image/png;base64,{base64.b64encode(imageData.tobytes()).decode('utf-8')}"
                tcb.inspectRecord = finalImage
                resultData = {
                    'status': 'creating',
                    'progress': progress,
                    'tempImage': encodedImageData,
                    'tempId': tcb.taskId
                }
                return jsonify({'success': True, 'result': resultData})
            elif isinstance(tcb, VideoTCB):
                with executorLock:
                    progress = str(int(executor.getCurIter() / tcb.iteration * 100))
                    executor.queryTime = time.time()
                return jsonify({'success': True, 'result': {'status': 'creating', 'progress': progress}})
            else:
                return jsonify({'success': False, 'result': 'Unknown task type.'})
        elif status == QueryStatus.WAITING_TASK:
            # 生成任务正在等待
            with executorLock:
                executor.queryTime = time.time()
            return jsonify({'success': True, 'result': {'status': 'waiting'}})
        elif status == QueryStatus.ABNORMAL_TASK:
            # 生成任务发生异常
            exceptionQueue.remove(tcb)
            return jsonify({'success': True, 'result': {'status': 'abnormal'}})
        elif status == QueryStatus.NO_TASK:
            # 未找到生成任务
            return jsonify({'success': True, 'result': {'status': 'none'}})
        else:
            # 错误生成任务查询状态
            return jsonify({'success': False, 'result': 'Wrong task status.'})
    except Exception as e:
        return jsonify({'success': False, 'result': e})


# 提交文本生成任务
@blue_painter.route('/creator/submitTextTask', methods=['POST'])
def submitTextTask():
    try:
        text = request.form.get('text', '').strip()
        wordLimit = str(int(int(request.form.get('wordLimit')) * 0.8))
        textType = request.form.get('textType')
        prompt = request.form.get('prompt', '').strip()
        if textType == 'cueWord':
            case = "“湛蓝天空下，金黄稻田随风起伏，远处农舍炊烟袅袅，画面充满乡村宁静气息”，“繁华都市的霓虹灯闪烁，高楼大厦林立，车水马龙的街道上人群熙熙攘攘，展现出城市的喧嚣与活力”，“神秘的古代城堡坐落在幽暗森林深处，月光透过树枝洒在城堡斑驳的墙壁上，增添了几分诡异氛围”，“清澈湖水如镜面般倒映着蓝天白云，湖边开满五颜六色的野花，湖面上有几只白天鹅优雅游弋”，“太空站漂浮在浩瀚宇宙中，周围是璀璨的星星和绚丽的星云，科技感十足”，“可爱小女孩在开满蒲公英的草地上奔跑，阳光洒在她笑脸上，蒲公英种子随风飘散，画面洋溢着童真和快乐”"
            if text == '':
                if prompt == '':
                    finalPrompt = "请你模仿以下句子的风格，生成一段场景描述，字数限制在" + wordLimit + "个字以内：" + case
                else:
                    finalPrompt = "请你模仿以下句子的风格，以“" + prompt + "”为主题或内容，生成或提炼出一段场景描述，字数限制在" + wordLimit + "个字以内：" + case
            else:
                finalPrompt = "请你模仿" + case + "等句子的风格，将以下句子或单词润色或连词成句，字数限制在" + wordLimit + "个字以内：" + text
        elif textType == 'theme':
            case = "“落日余晖洒满湖面，波光粼粼，一对恋人相拥而坐，四周花香弥漫，浪漫至极”，“晨曦微露，薄雾笼罩山林，鸟儿啁啾，溪水潺潺，山花烂漫，宛如世外桃源”，“幽暗洞穴内荧光水滴垂下水晶闪烁低语声声”，“老旧火车站蒸汽火车喷白烟月台人群别离”，“文人书房，笔墨纸砚，窗含竹影，雅韵悠长”，“海底王国，珊瑚宫殿，人鱼游弋，宝珠闪耀”，“雪覆古堡，少女捧火种，红斗篷在风中飘扬，眼中满是绮丽幻想”，“徽派建筑群，白墙黛瓦，马头墙高耸。庭院里老人摇蒲扇，轻声诉说着往昔故事”"
            if text == '':
                if prompt == '':
                    finalPrompt = "请你模仿以下句子的风格，生成一段主题，字数限制在" + wordLimit + "个字以内：" + case
                else:
                    finalPrompt = "请你模仿以下句子的风格，以“" + prompt + "”为内容，生成或提炼出一段主题，字数限制在" + wordLimit + "个字以内：" + case
            else:
                finalPrompt = "请你模仿" + case + "等句子的风格，将以下句子或单词润色或连词成句以生成一段主题，字数限制在" + wordLimit + "个字以内：" + text
        elif textType == 'content':
            case = "“画面缓缓展开，光影交织间，似藏着无尽故事。那柔和色调，仿若春日暖阳轻抚心田，温暖且熨帖；又恰似微风拂过湖面，泛起层层细腻涟漪，于静谧中生出灵动韵律。它宛如灵动的生命，在光影的舞台上翩然起舞，举手投足间尽显自然灵秀；又仿若穿越时空的低语，带着岁月的沉香与温度，诉说着那些被时光冲刷却愈发醇厚的过往，每一道细微的纹理、每一抹浅淡的色彩，皆是岁月馈赠的诗行，在最平凡的瞬间捕捉到那一抹不平凡，让观者沉浸其中，回味无穷，于平淡日常里发现熠熠生辉的美好”"
            if text == '':
                if prompt == '':
                    finalPrompt = "请你模仿以下句子的风格，生成一段对于画面或图画的内容描述，字数限制在" + wordLimit + "个字以内：" + case
                else:
                    finalPrompt = "请你模仿以下句子的风格，以“" + prompt + "”为内容或需要描述的对象，生成一段描述，字数限制在" + wordLimit + "个字以内：" + case
            else:
                finalPrompt = "请你模仿" + case + "等句子的风格，将以下句子或单词润色或连词成句以生成一段对于画面或图画的内容描述，字数限制在" + wordLimit + "个字以内：" + text
        else:
            return jsonify({'success': False, 'result': 'Wrong text generation type.'})
        reply = ollama.generate(model='deepseek-r1:7b', prompt=finalPrompt)
        finalReply = re.sub(r'<think>.*?</think>', '', reply['response'], flags=re.DOTALL).replace('\n', '')
        return jsonify({'success': True, 'result': finalReply})
    except Exception as e:
        return jsonify({'success': False, 'result': e})


if __name__ == '__main__':
    app.register_blueprint(blue_painter)
    app.run(host='0.0.0.0', port=5000)
