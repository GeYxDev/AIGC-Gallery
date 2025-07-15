// 引入事件总线
import EventBus from '../../../utils/event-bus'

Component({
  properties: {
    imageId: {
      type: String,
      value: ''
    },
    sourcePageId: {
      type: String,
      value: ''
    },
    mediaList: {
      type: Array,
      value: []
    },
    mediaType: {
      type: String,
      value: ''
    }
  },
  data: {
    mediaWidth: 0,
    mediaHeight: 0,
    index: 0,
    tempIndex: -1
  },
  observers: {
    tempIndex(tempIndex) {
      // 切换图片时传递变化至前一个页面
      const { mediaList, sourcePageId } = this.data;
      const image = mediaList[tempIndex];
      if (image) EventBus.emit(`${sourcePageId}PreviewerChange`, image);
    }
  },
  lifetimes: {
    attached() {
      // 计算视频显示大小
      if (this.data.mediaType === 'video') {
        const windowWidth = getApp().globalData.windowWidth
        const windowHeight = getApp().globalData.windowHeight
        const windowRatio = windowWidth / windowHeight
        if (this.data.mediaList[0].mediaRatio > windowRatio) {
          // 视频的宽高比大于屏幕的宽高比时以屏幕宽度为基准
          this.setData({
            'mediaWidth': windowWidth,
            'mediaHeight': windowWidth / this.data.mediaList[0].mediaRatio
          })
        } else {
          // 视频的宽高比小于屏幕的宽高比时以屏幕高度为基准
          this.setData({
            'mediaWidth': windowHeight * this.data.mediaList[0].mediaRatio,
            'mediaHeight': windowHeight
          })
        }
      }
      // 获得当前图片标识符
      const imageId = this.data.imageId
      // 获得媒体列表
      const mediaList = this.data.mediaList
      // 设置当前图片索引
      let index = 0
      if (imageId) {
        const currentIndex = mediaList.findIndex(item => item.id.toString() === imageId)
        if (currentIndex !== -1) index = currentIndex
      }
      this.setData({ index })
    },
    detached() {
      // 通知页面媒体预览将被销毁
      EventBus.emit(`${this.data.sourcePageId}PreviewerDestroy`)
    }
  },
  methods: {
    // 下层触发事件处理
    onBeforeRender(e) {
      // 下层媒体切换索引信息
      const { index } = e.detail;
      this.data.index = index;
      this.setData({ tempIndex: index });
    }
  }
})