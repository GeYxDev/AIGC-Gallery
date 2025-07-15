// 引入动画飞行状态
import { FlightDirection } from '../../utils/route'

Component({
  options: {
    virtualHost: true
  },
  properties: {
    // 组件索引以及ID
    index: {
      type: Number,
      value: -1
    },
    // 传入显示数据
    item: {
      type: Object,
      value: {}
    },
    // 预设卡片宽度
    cardWidth: {
      type: Number,
      value: 0
    }
  },
  lifetimes: {
    created() {
      this.scale = wx.worklet.shared(1)
      this.opacity = wx.worklet.shared(0)
      this.direction = wx.worklet.shared(0)
      this.srcWidth = wx.worklet.shared('100%')
      this.radius = wx.worklet.shared(5)
      const beginRect = wx.worklet.shared(undefined)
      const endRect = wx.worklet.shared(undefined)
      // 在JS和UI线程之间共享变量
      wx.worklet.runOnUI(() => {
        'worklet'
        globalThis['RouteCardSrcRect'] = beginRect
        globalThis['RouteCardDestRect'] = endRect
      })()
    },
    attached() {
      // 卡片显示部分过渡动画
      this.applyAnimatedStyle(
        '.card-display', 
        () => {
          'worklet'
          return {
            width: this.srcWidth.value,
            transform: `scale(${this.scale.value})`
          }
        }, 
        {
          immediate: false,
          flush: 'sync'
        },
        () => {}
      )
      // 媒体显示过渡动画
      this.applyAnimatedStyle(
        '.card-media',
        () => {
          'worklet'
          return {
            opacity: this.opacity.value,
            borderTopRightRadius: this.radius.value,
            borderTopLeftRadius: this.radius.value,
          }
        },
        {
          immediate: false,
          flush: 'sync'
        },
        () => {}
      )
      // 描述区过渡动画
      this.applyAnimatedStyle(
        '.card-desc',
        () => {
          'worklet'
          return {
            opacity: this.opacity.value,
          }
        },
        {
          immediate: false,
          flush: 'sync'
        },
        () => {}
      )
    }
  },
  methods: {
    // 卡片跳转至细节页面
    jumpToDetail() {
      const { groupId, account, nickname, avatar, type, media, theme, aspectRatio, likes } = this.data.item
      const urlContent = `/pages/detail/detail?index=${this.data.index}&groupId=${groupId}&account=${account}&nickname=${nickname}&avatar=${avatar}&type=${type}&media=${media}&theme=${theme}&aspectRatio=${aspectRatio}&likes=${likes}`
      wx.navigateTo({
        url: urlContent,
        routeType: 'cardScaleTransition'
      })
    },
    // 处理过渡动画帧
    handleGradFrame(data) {
      'worklet'
      this.direction.value = data.direction
      if (data.direction === FlightDirection.PUSH) {
        // card -> detail
        this.srcWidth.value = `${data.begin.width}px`
        this.scale.value = data.current.width / data.begin.width
        this.opacity.value = 1 - data.progress
        this.radius.value = 0
      } else if (data.direction === FlightDirection.POP) {
        // detail -> card
        this.scale.value = data.current.width / data.end.width
        this.opacity.value = data.progress
        this.radius.value = 5
      }
      // 赋值用于其他页面访问动画起始与结束信息
      if (globalThis['RouteCardSrcRect'] && globalThis['RouteCardSrcRect'].value == undefined) {
        globalThis['RouteCardSrcRect'].value = data.begin
      }
      if (globalThis['RouteCardDestRect'] && globalThis['RouteCardDestRect'].value == undefined) {
        globalThis['RouteCardDestRect'].value = data.end
      }
    }
  }
});