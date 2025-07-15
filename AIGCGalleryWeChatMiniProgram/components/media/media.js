// 引入事件总线
import EventBus from '../../utils/event-bus'

// 变量与索引对应关系
const IMAGE_INIT_WIDTH = 0
const IMAGE_INIT_HEIGHT = 1
const IMAGE_WIDTH = 2
const IMAGE_HEIGHT = 3
const IMAGE_TARGET_WIDTH = 4
const IMAGE_TARGET_HEIGHT = 5
const IMAGE_RATIO = 6
const IN_PREVIEW = 7

Component({
  properties: {
    workId: {
      type: Number,
      value: -1
    },
    media: {
      type: String,
      value: ''
    },
    mediaWidth: {
      type: Number,
      value: 0
    },
    mediaHeight: {
      type: Number,
      value: 0
    }
  },
  lifetimes: {
    created() {
      // 共享状态用于存储动画相关状态值
      this.sharedValues = [
        wx.worklet.shared(0), // IMAGE_INIT_WIDTH
        wx.worklet.shared(0), // IMAGE_INIT_HEIGHT
        wx.worklet.shared(0), // IMAGE_WIDTH
        wx.worklet.shared(0), // IMAGE_HEIGHT
        wx.worklet.shared(0), // IMAGE_TARGET_WIDTH
        wx.worklet.shared(0), // IMAGE_TARGET_HEIGHT
        wx.worklet.shared(0), // IMAGE_RATIO
        wx.worklet.shared(false) // IN_PREVIEW
      ]
    },
    attached() {
      // 获取页面唯一标识符
      const pageId = this.getPageId()
      // 创建媒体唯一标识符
      const uniqueId = `${pageId}-${this.data.workId}`
      // 获取共享状态
      const sharedValues = this.sharedValues ?? []
      // 将媒体唯一标识符存储至组件实例
      this.uniqueId = uniqueId
      // 绑定动画至媒体
      wx.worklet.runOnUI(() => {
        'worklet';
        // 未注册回调函数则进行注册
        if (!globalThis.temp[`${uniqueId}CustomRouteBack`]) {
          globalThis.temp[`${uniqueId}CustomRouteBack`] = args => {
            if (!sharedValues[IN_PREVIEW].value) return;
            // 对宽高进行矫正以获得正确缩放
            const targetImageWidth = sharedValues[IMAGE_TARGET_WIDTH].value;
            const targetImageHeight = sharedValues[IMAGE_TARGET_HEIGHT].value;
            const scale = args.scale;
            sharedValues[IMAGE_WIDTH].value = targetImageWidth * scale;
            sharedValues[IMAGE_HEIGHT].value = targetImageHeight * scale;
          }
          // 网页拖动时产生动画效果
          globalThis.eventBus.on(`${pageId}CustomRouteBack`, globalThis.temp[`${uniqueId}CustomRouteBack`]);
        }
      })()
      // 应用动画至图片或视频封面
      this.applyAnimatedStyle('.media-image', () => {
        'worklet';
        let width = `${sharedValues[IMAGE_WIDTH].value}px`;
        if (sharedValues[IMAGE_WIDTH].value === 0) width = ``;
        let height = `${sharedValues[IMAGE_HEIGHT].value}px`;
        if (sharedValues[IMAGE_HEIGHT].value === 0) height = ``;
        return { width, height };
      })
      // 监听预览页面媒体切换事件
      this._onPreviewerChange = image => {
        if (image.workId === this.data.workId) {
          sharedValues[IN_PREVIEW].value = true;
        } else {
          resetShareValues();
        }
      }
      EventBus.on(`${pageId}PreviewerChange`, this._onPreviewerChange)
      // 共享状态中状态值重置
      const resetShareValues = () => {
        sharedValues[IMAGE_WIDTH].value = sharedValues[IMAGE_INIT_WIDTH].value;
        sharedValues[IMAGE_HEIGHT].value = sharedValues[IMAGE_INIT_HEIGHT].value;
        sharedValues[IN_PREVIEW].value = false;
      }
      // 预览页销毁时恢复媒体预览原始状态
      this._onPreviewerHide = () => {
        setTimeout(resetShareValues, 500);
      }
      EventBus.on(`${pageId}PreviewerDestroy`, this._onPreviewerHide)
    },
    detached() {
      // 获取页面唯一标识符
      const pageId = this.getPageId()
      // 获取媒体唯一标识符
      const uniqueId = this.uniqueId
      // 清除绑定在媒体上的动画
      wx.worklet.runOnUI(() => {
        'worklet';
        if (globalThis.temp[`${uniqueId}CustomRouteBack`]) {
          globalThis.eventBus.off(`${pageId}CustomRouteBack`, globalThis.temp[`${uniqueId}CustomRouteBack`]);
          delete globalThis.temp[`${uniqueId}CustomRouteBack`];
        }
      })()
      // 移除图片切换监听器
      EventBus.off(`${pageId}PreviewerChange`, this._onPreviewerChange)
      // 移除页面销毁监听器
      EventBus.off(`${pageId}PreviewerDestroy`, this._onPreviewerHide)
    }
  },
  methods: {
    // 处理过渡动画帧
    handleGradFrame(e) {
      'worklet';
      // 根据媒体预览大小调整图片或视频封面大小
      const rect = e.current;
      const sharedValues = this.sharedValues ?? [];
      const cntWidth = rect.width;
      const cntHeight = rect.height;
      // 当前动画进度
      const progress = e.progress;
      const imageRatio = sharedValues[IMAGE_RATIO].value;
      const isPop = e.direction === 1;
      let width = cntWidth;
      let height = cntHeight;
      if (imageRatio) {
        const cntRatio = cntWidth / cntHeight;
        if (cntRatio > imageRatio) height = cntWidth / imageRatio;
        else if (cntRatio <= imageRatio) width = cntHeight * imageRatio;
        // 获取媒体的初始大小和目标大小
        const initImageWidth = sharedValues[IMAGE_INIT_WIDTH].value;
        const initImageHeight = sharedValues[IMAGE_INIT_HEIGHT].value;
        const targetImageWidth = sharedValues[IMAGE_TARGET_WIDTH].value;
        const targetImageHeight = sharedValues[IMAGE_TARGET_HEIGHT].value;
        if (initImageWidth && initImageHeight && targetImageWidth && targetImageHeight) {
          if (isPop) {
            // 执行退出动画
            width = targetImageWidth - (targetImageWidth - initImageWidth) * progress;
            height = targetImageHeight - (targetImageHeight - initImageHeight) * progress;
          } else {
            // 执行进入动画
            width = initImageWidth + (targetImageWidth - initImageWidth) * progress;
            height = initImageHeight + (targetImageHeight - initImageHeight) * progress;
          }
        }
      }
      sharedValues[IMAGE_WIDTH].value = width;
      sharedValues[IMAGE_HEIGHT].value = height;
    },
    // 媒体加载完成回调
    mediaLoadFinishCallback(e) {
      // 获得媒体实际大小
      const { width, height } = e.detail;
      const sharedValues = this.sharedValues ?? [];
      // 计算媒体比例
      const imageRatio = width / height;
      this.imageRatio = imageRatio;
      sharedValues[IMAGE_RATIO].value = imageRatio;
      this.initAnimationData();
    },
    // 初始化图像宽高与动画目标值
    initAnimationData() {
      const sharedValues = this.sharedValues ?? [];
      const { mediaWidth: cntWidth, mediaHeight: cntHeight } = this.data;
      const imageRatio = this.imageRatio;
      const windowWidth = getApp().globalData.windowWidth;
      const windowHeight = getApp().globalData.windowHeight;
      if (cntWidth && cntHeight && imageRatio) {
        let initWidth = cntWidth;
        let initHeight = cntHeight;
        let targetImageWidth = windowWidth;
        let targetImageHeight = windowHeight;
        const cntRatio = cntWidth / cntHeight;
        const targetRatio = windowWidth / windowHeight;
        // 根据预览显示的媒体比例与媒体实际比例决定初始大小
        if (cntRatio > imageRatio) {
          initHeight = cntWidth / imageRatio;
        } else if (cntRatio < imageRatio) {
          initWidth = cntHeight * imageRatio;
        }
        // 根据媒体实际比例与窗口显示比例决定目标媒体大小
        if (targetRatio > imageRatio) {
          targetImageWidth = targetImageHeight * imageRatio;
        } else if (targetRatio < imageRatio) {
          targetImageHeight = targetImageWidth / imageRatio;
        }
        // 在关注圈页面时的媒体初始大小
        sharedValues[IMAGE_INIT_WIDTH].value = initWidth;
        sharedValues[IMAGE_INIT_HEIGHT].value = initHeight;
        // 预览时媒体显示大小
        sharedValues[IMAGE_WIDTH].value = initWidth;
        sharedValues[IMAGE_HEIGHT].value = initHeight;
        // 浏览时媒体显示目标大小
        sharedValues[IMAGE_TARGET_WIDTH].value = targetImageWidth;
        sharedValues[IMAGE_TARGET_HEIGHT].value = targetImageHeight;
      }
    }
  }
})