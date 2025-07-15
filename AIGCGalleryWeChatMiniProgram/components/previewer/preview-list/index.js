// 引入手势与动画相关方法
import { PreviewerGesture } from '../../../utils/route'

// 变量与索引对应关系
const GESTURE_STATE = 0
const CURRENT_ID = 1

Component({
  properties: {
    index: {
      type: Number,
      value: 0
    },
    mediaList: {
      type: Array,
      value: []
    }
  },
  data: {
    currentIndex: -100,
    needSwiperAnimation: true
  },
  observers: {
    mediaList(mediaList) {
      // 媒体列表更新时更新媒体索引唯一值并切换图片
      const index = this.data.index;
      const current = mediaList[index];
      if (!current) return;
      const sharedValues = this.sharedValues ?? [];
      sharedValues[CURRENT_ID].value = current.id;
      this.toggleImage(index, true);
    },
    // 图片索引更新时切换图片
    index(index) {
      const len = this.data.mediaList.length;
      if (!len) return;
      if (index !== this.data.currentIndex && index >= 0 && index < len) {
        this.toggleImage(index, true);
      } else {
        const image = this.data.mediaList[index];
        if (image && image.id === this.currentRenderImage) {
          this.onImageRender({ detail: image });
        }
      }
    }
  },
  lifetimes: {
    created() {
      // 共享状态用于存储动画相关状态值
      this.sharedValues = [
        wx.worklet.shared(0), // GESTURE_STATE
        wx.worklet.shared('') // CURRENT_ID
      ]
    },
    async attached() {
      // 获得页面唯一索引值
      const pageId = this.getPageId()
      // 获取共享状态
      const sharedValues = this.sharedValues ?? []
      // 绑定媒体预览列表动画
      wx.worklet.runOnUI(() => {
        'worklet';
        // 监听拖拽返回手势
        if (!globalThis.temp[`${pageId}PreviewerBack`]) {
          globalThis.temp[`${pageId}PreviewerBack`] = () => sharedValues[GESTURE_STATE].value = PreviewerGesture.Back;
          globalThis.eventBus.on(`${pageId}Back`, globalThis.temp[`${pageId}PreviewerBack`]);
        }
        // 监听图片切换手势
        if (!globalThis.temp[`${pageId}PreviewerToggle`]) {
          globalThis.temp[`${pageId}PreviewerToggle`] = () => sharedValues[GESTURE_STATE].value = PreviewerGesture.Toggle;
          globalThis.eventBus.on(`${pageId}Toggle`, globalThis.temp[`${pageId}PreviewerToggle`]);
        }
        // 监听图片拖动中事件
        if (!globalThis.temp[`${pageId}PreviewerMoving`]) {
          globalThis.temp[`${pageId}PreviewerMoving`] = () => sharedValues[GESTURE_STATE].value = PreviewerGesture.Moving;
          globalThis.eventBus.on(`${pageId}Moving`, globalThis.temp[`${pageId}PreviewerMoving`]);
        }
        // 监听图片拖动手势
        if (!globalThis.temp[`${pageId}PreviewerMove`]) {
          globalThis.temp[`${pageId}PreviewerMove`] = args => {
            const currentId = sharedValues[CURRENT_ID].value;
            globalThis.eventBus.emit(`${pageId}${currentId}Move`, args);
          }
          globalThis.eventBus.on(`${pageId}Move`, globalThis.temp[`${pageId}PreviewerMove`]);
        }
        // 监听图片放缩手势
        if (!globalThis.temp[`${pageId}PreviewerScale`]) {
          globalThis.temp[`${pageId}PreviewerScale`] = args => {
            const currentId = sharedValues[CURRENT_ID].value;
            globalThis.eventBus.emit(`${pageId}${currentId}Scale`, args);
          }
          globalThis.eventBus.on(`${pageId}Scale`, globalThis.temp[`${pageId}PreviewerScale`]);
        }
        // 监听手势结束事件
        if (!globalThis.temp[`${pageId}PreviewerEnd`]) {
          globalThis.temp[`${pageId}PreviewerEnd`] = args => {
            const currentId = sharedValues[CURRENT_ID].value;
            globalThis.eventBus.emit(`${pageId}${currentId}End`, args);
          }
          globalThis.eventBus.on(`${pageId}End`, globalThis.temp[`${pageId}PreviewerEnd`]);
        }
      })()
    },
    detached() {
      // 获得页面唯一标识符
      const pageId = this.getPageId()
      // 取消媒体预览列表动画
      wx.worklet.runOnUI(() => {
        'worklet';
        const removeList = ['Back', 'Toggle', 'Moving', 'Move', 'Scale', 'End'];
        removeList.forEach(item => {
          'worklet';
          const globalKey = `${pageId}Previewer${item}`;
          if (globalThis.temp[globalKey]) {
            globalThis.eventBus.off(`${pageId}${item}`, globalThis.temp[globalKey]);
            delete globalThis.temp[globalKey];
          }
        })
      })()
    }
  },
  methods: {
    // 更新图片唯一标识符和索引，并通知上层目前图片状态
    async toggleImage(index, disableAnimation = false) {
      const image = this.data.mediaList[index];
      if (!image) return;
      const sharedValues = this.sharedValues ?? [];
      sharedValues[CURRENT_ID].value = image.id;
      this.setData({
        currentIndex: index,
        needSwiperAnimation: !disableAnimation
      });
      this.data.index = index;
      this.triggerEvent('beforerender', { index, image });
    },
    // 图片渲染完成时更新当前图片唯一标识符
    onImageRender(e) {
      const mediaList = this.data.mediaList;
      const index = this.data.currentIndex;
      const image = mediaList[index] || {};
      if (e.detail.id !== image.id) return;
      this.currentRenderImage = image.id;
    },
    // 媒体点击处理
    onTapImage() {
      'worklet';
      wx.worklet.runOnJS(this.triggerEvent.bind(this))('tapimage');
    },
    // 响应移动判断
    shouldResponseOnMove() {
      'worklet';
      const sharedValues = this.sharedValues ?? [];
      return sharedValues[GESTURE_STATE].value === PreviewerGesture.Toggle;
    },
    // 图片滑动切换索引更新
    onSwiperChange(e) {
      const { current, source } = e.detail;
      if (source === 'touch') this.toggleImage(current, false);
    }
  }
})