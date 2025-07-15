// 引入事件总线
import EventBus from '../../utils/event-bus'
// 引入手势与动画相关方法
import { PreviewerGesture, AnimationStatus, GestureState, recoverTiming, calcOpacity, calcScale } from '../../utils/route'

// 变量与索引对应关系
const TRANSLATE_X = 0
const TRANSLATE_Y = 1
const START_Y = 2
const OPACITY = 3
const SCALE = 4
const MIN_SCALE = 5
const USER_GESTURE_IN_PROGRESS = 6
const GESTURE_STATE = 7
const PAGE_ID = 8
const TEMP_LAST_SCALE = 9

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
  lifetimes: {
    created() {
      // 初始化事件总线
      EventBus.initWorkletEventBus()
      // 共享状态用于存储动画相关状态值
      this.sharedValues = [
        wx.worklet.shared(0), // TRANSLATE_X
        wx.worklet.shared(0), // TRANSLATE_Y
        wx.worklet.shared(0), // START_Y
        wx.worklet.shared(1), // OPACITY
        wx.worklet.shared(1), // SCALE
        wx.worklet.shared(1), // MIN_SCALE
        wx.worklet.shared(false), // USER_GESTURE_IN_PROGRESS
        wx.worklet.shared(0), // GESTURE_STATE
        wx.worklet.shared(0), // PAGE_ID
        wx.worklet.shared(1) // TEMP_LAST_SCALE
      ]
    },
    attached() {
      // 获得原页面唯一标识符
      const sourcePageId = this.data.sourcePageId
      // 获得屏幕高度
      const windowHeight = getApp().globalData.windowHeight
      // 获得页面唯一标识符
      const pageId = this.getPageId()
      // 获取共享状态
      const sharedValues = this.sharedValues ?? []
      // 存储当前页面唯一标识符
      sharedValues[PAGE_ID].value = pageId
      // 获得路由上下文
      this.customRouteContext = wx.router.getRouteContext(this)
      // 绑定预览动画
      wx.worklet.runOnUI(() => {
        'worklet';
        // 监听拖拽返回手势
        if (!globalThis.temp[`${pageId}GestureBack`]) {
          globalThis.temp[`${pageId}GestureBack`] = args => {
            if (sharedValues[GESTURE_STATE].value === PreviewerGesture.Moving) return;
            sharedValues[GESTURE_STATE].value = PreviewerGesture.Back;
            const { moveX, moveY, offsetY } = args;
            // 横向滑动跟随手势
            sharedValues[TRANSLATE_X].value += moveX;
            // 竖向手势上滑时存在阻尼
            if (sharedValues[TRANSLATE_Y].value < 0) {
              const fy = 0.52 * ((1 - Math.min(offsetY / windowHeight, 1)) ** 2);
              const translateY = sharedValues[TRANSLATE_Y].value + Math.ceil(moveY * fy);
              sharedValues[TRANSLATE_Y].value = translateY;
            } else {
              sharedValues[TRANSLATE_Y].value += moveY;
            }
            // 手势拖动产生渐变
            sharedValues[OPACITY].value = calcOpacity(offsetY, windowHeight);
            // 手势拖动产生大小变化
            const scale = calcScale(offsetY, windowHeight);
            sharedValues[SCALE].value = scale;
            sharedValues[MIN_SCALE].value = Math.min(scale, sharedValues[MIN_SCALE].value);
          }
          globalThis.eventBus.on(`${pageId}Back`, globalThis.temp[`${pageId}GestureBack`]);
        }
        // 监听拖拽返回手势结束
        if (!globalThis.temp[`${pageId}GestureBackEnd`]) {
          globalThis.temp[`${pageId}GestureBackEnd`] = () => {
            const moveY = sharedValues[TRANSLATE_Y].value;
            const scale = sharedValues[SCALE].value;
            const minScale = sharedValues[MIN_SCALE].value;
            const { didPop } = this.customRouteContext || {};
            if (moveY > 1 && scale <= (minScale + 0.01)) {
              // 一直处于缩小状态时退出页面，否则恢复页面
              globalThis.eventBus.emit(`${sourcePageId}CustomRouteBack`, { scale });
              didPop();
            } else {
              sharedValues[OPACITY].value = recoverTiming(1);
              sharedValues[TRANSLATE_X].value = recoverTiming(0);
              sharedValues[TRANSLATE_Y].value = recoverTiming(0);
              sharedValues[SCALE].value = recoverTiming(1, () => {
                'worklet';
                sharedValues[GESTURE_STATE].value = PreviewerGesture.Init;
              })
              sharedValues[MIN_SCALE].value = 1;
            }
          }
          globalThis.eventBus.on(`${pageId}BackEnd`, globalThis.temp[`${pageId}GestureBackEnd`]);
        }
        // 监听图片切换手势
        if (!globalThis.temp[`${pageId}GestureToggle`]) {
          globalThis.temp[`${pageId}GestureToggle`] = () => {
            if (sharedValues[GESTURE_STATE].value === PreviewerGesture.Moving) return;
            sharedValues[GESTURE_STATE].value = PreviewerGesture.Toggle;
          }
          globalThis.eventBus.on(`${pageId}Toggle`, globalThis.temp[`${pageId}GestureToggle`]);
        }
        // 监听图片拖动手势
        if (!globalThis.temp[`${pageId}GestureMoving`]) {
          globalThis.temp[`${pageId}GestureMoving`] = () => {
            sharedValues[GESTURE_STATE].value = PreviewerGesture.Moving;
          }
          globalThis.eventBus.on(`${pageId}Moving`, globalThis.temp[`${pageId}GestureMoving`]);
        }
      })()
      // 为共享动画元素添加动画，实现页面穿梭
      this.applyAnimatedStyle('#preview-home >>> .need-transform-on-back', () => {
        'worklet';
        if (this.data.mediaType === 'image') {
          return {
            transform: `translate(${sharedValues[TRANSLATE_X].value}px, ${sharedValues[TRANSLATE_Y].value}px) scale(${sharedValues[SCALE].value})`
          };
        } else if (this.data.mediaType === 'video') {
          return {
            transform: `translate(${sharedValues[TRANSLATE_X].value}px, ${sharedValues[TRANSLATE_Y].value}px) scale(${sharedValues[SCALE].value})`,
            opacity: Math.max(0, Math.min(1, 1 - 0.2 * Math.pow(1 - sharedValues[OPACITY].value, 2)))
          };
        }
      })
      const { primaryAnimation, primaryAnimationStatus } = this.customRouteContext || {}
      // 为媒体预览添加动画，实现隐藏
      this.applyAnimatedStyle('#preview-home >>> .need-hide-on-back', () => {
        'worklet';
        const status = primaryAnimationStatus.value;
        const isRunningAnimation = status === 1 || status === 2;
        return {
          left: (isRunningAnimation || (sharedValues[GESTURE_STATE].value === PreviewerGesture.Back)) ? '9999px' : '0'
        };
      })
      // 为媒体背景添加动画，实现渐变淡出
      this.applyAnimatedStyle('#preview-home >>> .preview-middle-self', () => {
        'worklet';
        const status = primaryAnimationStatus.value;
        const opacity = sharedValues[OPACITY].value;
        if (!sharedValues[USER_GESTURE_IN_PROGRESS].value) {
          // 非手势触发时加速动画
          const value = primaryAnimation.value;
          let factor = value;
          if (status === AnimationStatus.forward) {
            factor *= 3;
            if (factor > 1) factor = 1;
          } else if (status === AnimationStatus.reverse) {
            factor = 1 - ((1 - factor) * 3);
            if (factor < 0) factor = 0;
          }
          const newOpacity = opacity * factor;
          return { opacity: newOpacity > 1 ? 1 : newOpacity };
        } else {
          // 手势触发时正常动画过程
          return { opacity };
        }
      })
    },
    detached() {
      // 获取页面唯一标识符
      const pageId = this.getPageId();
      // 取消预览动画
      wx.worklet.runOnUI(() => {
        'worklet';
        const removeList = ['Back', 'BackEnd', 'Toggle', 'Moving'];
        removeList.forEach(item => {
          'worklet';
          const globalKey = `${pageId}Gesture${item}`;
          if (globalThis.temp[globalKey]) {
            globalThis.eventBus.off(`${pageId}${item}`, globalThis.temp[globalKey]);
            delete globalThis.temp[globalKey];
          }
        })
      })()
    }
  },
  methods: {
    // 缩放手势控制
    onScale(e) {
      'worklet';
      const sharedValues = this.sharedValues ?? [];
      const pageId = sharedValues[PAGE_ID].value;
      if (e.state === GestureState.BEGIN) {
        sharedValues[START_Y].value = e.focalY;
        sharedValues[TEMP_LAST_SCALE].value = 1;
        sharedValues[GESTURE_STATE].value = PreviewerGesture.Init;
        sharedValues[USER_GESTURE_IN_PROGRESS].value = true;
      } else if (e.state === GestureState.ACTIVE) {
        const focalX = e.focalX;
        const focalY = e.focalY;
        const moveX = e.focalDeltaX;
        const moveY = e.focalDeltaY;
        const offsetY = focalY - sharedValues[START_Y].value;
        if (e.pointerCount === 2) {
          // 双指缩放手势
          const pageId = sharedValues[PAGE_ID].value;
          const realScale = e.scale / sharedValues[TEMP_LAST_SCALE].value;
          sharedValues[TEMP_LAST_SCALE].value = e.scale;
          globalThis.eventBus.emit(`${pageId}Scale`, { scale: realScale, centerX: focalX, centerY: focalY });
        } else if (sharedValues[GESTURE_STATE].value === PreviewerGesture.Back) {
          // 拖拽返回手势
          globalThis.eventBus.emit(`${pageId}Back`, { moveX, moveY, offsetY });
        } else if (sharedValues[GESTURE_STATE].value === PreviewerGesture.Toggle) {
          // 图片切换手势，不做处理
        } else {
          // 单指图片拖动手势
          globalThis.eventBus.emit(`${pageId}Move`, { moveX, moveY, offsetY, origin: 'move'});
        }
      } else if (e.state === GestureState.END || e.state === GestureState.CANCELLED) {
        const velocityX = e.velocityX;
        const velocityY = e.velocityY;
        sharedValues[USER_GESTURE_IN_PROGRESS].value = false;
        if (sharedValues[GESTURE_STATE].value === PreviewerGesture.Back) {
          // 拖拽返回手势结束
          globalThis.eventBus.emit(`${pageId}BackEnd`);
        } else if (sharedValues[GESTURE_STATE].value === PreviewerGesture.Toggle) {
          // 切换图片手势结束
          sharedValues[GESTURE_STATE].value = PreviewerGesture.Init;
        } else {
          // 其他手势结束
          globalThis.eventBus.emit(`${pageId}End`, { velocityX, velocityY });
          sharedValues[GESTURE_STATE].value = PreviewerGesture.Init;
        }
      }
    },
    // 响应移动判断
    shouldResponseOnMove() {
      'worklet';
      return true;
    }
  }
})