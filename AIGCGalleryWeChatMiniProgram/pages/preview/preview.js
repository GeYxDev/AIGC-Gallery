Component({
  properties: {
    // 媒体唯一标识符
    rawImageId: {
      type: String,
      value: ''
    },
    // 原页面唯一标识符
    rawSourcePageId: {
      type: String,
      value: ''
    },
    // 媒体列表
    rawMediaList: {
      type: String,
      value: ''
    },
    // 媒体类型
    rawMediaType: {
      type: String,
      value: ''
    }
  },
  data: {
    imageId: '',
    sourcePageId: '',
    mediaList: [],
    mediaType: ''
  },
  lifetimes: {
    attached() {
      // 数据解码并赋值
      const imageId = decodeURIComponent(this.data.rawImageId)
      const sourcePageId = decodeURIComponent(this.data.rawSourcePageId)
      const mediaList = JSON.parse(decodeURIComponent(this.data.rawMediaList))
      const mediaType = decodeURIComponent(this.data.rawMediaType)
      this.setData({ imageId, sourcePageId, mediaList, mediaType })
    }
  }
})