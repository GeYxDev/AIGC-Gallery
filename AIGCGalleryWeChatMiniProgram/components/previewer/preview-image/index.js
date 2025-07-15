// 引入手势与动画相关方法
import { adjustTiming, recoverTiming } from '../../../utils/route'

// 变量与索引对应关系
const MOVE_X = 0
const MOVE_Y = 1
const SCALE = 2
const IMG_WIDTH = 3
const IMG_HEIGHT = 4
const IMG_MAX_SCALE = 5
const IMG_MIN_SCALE = 6
const IMG_MIN_MOVE_X = 7
const IMG_MAX_MOVE_X = 8
const IMG_MIN_MOVE_Y = 9
const IMG_MAX_MOVE_Y = 10
const GESTURE_STATE = 11
const TEMP_MOVE_X = 12
const TEMP_MOVE_Y = 13
const TEMP_SCALE = 14
const INIT_MOVE_X = 15
const INIT_MOVE_Y = 16
const INIT_SCALE = 17
const CURRENT_ID = 18
const PAGE_ID = 19
const DEFAULT_IMG_SCALE_MAX = 10

Component({
  properties: {
    status: {
      type: Number,
      value: 0
    },
    image: {
      type: Object,
      value: {}
    }
  },
  observers: {
    status(status) {
      // 当前显示状态变化时决定是否重置图片或触发渲染事件
      if (!this.isAttached) return;
      const oldStatus = this.oldStatus;
      if (status === oldStatus) return;
      this.oldStatus = status;
      if (status === 1) {
        if (this.isImgLoaded) this.triggerEvent('render', this.data.image);
      } else {
        this.resetImg();
      }
    }
  },
  lifetimes: {
    created() {
      // 共享状态用于存储动画相关状态值
      this.sharedValues = [
        wx.worklet.shared(0), // MOVE_X
        wx.worklet.shared(0), // MOVE_Y
        wx.worklet.shared(1), // SCALE
        wx.worklet.shared(0), // IMG_WIDTH
        wx.worklet.shared(0), // IMG_HEIGHT
        wx.worklet.shared(1), // IMG_MAX_SCALE
        wx.worklet.shared(1), // IMG_MIN_SCALE
        wx.worklet.shared(0), // IMG_MIN_MOVE_X
        wx.worklet.shared(0), // IMG_MAX_MOVE_X
        wx.worklet.shared(0), // IMG_MIN_MOVE_Y
        wx.worklet.shared(0), // IMG_MAX_MOVE_Y
        wx.worklet.shared(0), // GESTURE_STATE
        wx.worklet.shared(0), // TEMP_MOVE_X
        wx.worklet.shared(0), // TEMP_MOVE_Y
        wx.worklet.shared(1), // TEMP_SCALE
        wx.worklet.shared(0), // INIT_MOVE_X
        wx.worklet.shared(0), // INIT_MOVE_Y
        wx.worklet.shared(1), // INIT_SCALE
        wx.worklet.shared(''), // CURRENT_ID
        wx.worklet.shared('') // PAGE_ID
      ]
    },
    attached() {
      this.isAttached = true
      // 获得页面唯一标识符
      const pageId = this.getPageId()
      const windowWidth = getApp().globalData.windowWidth
      const windowHeight = getApp().globalData.windowHeight
      // 获得当前图片唯一标识符
      const id = this.data.image.id
      // 获取共享状态
      const sharedValues = this.sharedValues ?? []
      sharedValues[PAGE_ID].value = pageId
      sharedValues[CURRENT_ID].value = id
      // 应用放缩和拖动动画
      this.applyAnimatedStyle('#image', () => {
        'worklet';
        return {
          width: `${sharedValues[IMG_WIDTH].value}px`,
          height: `${sharedValues[IMG_HEIGHT].value}px`,
          transform: `translateX(${sharedValues[MOVE_X].value}px) translateY(${sharedValues[MOVE_Y].value}px) translateZ(0px) scale(${sharedValues[SCALE].value})`
        };
      })
      // 计算图片边缘
      function calcImgLimit(scale, imgWidth, imgHeight) {
        'worklet';
        const viewWidth = imgWidth * scale;
        const viewHeight = imgHeight * scale;
        if (viewWidth > windowWidth) {
          sharedValues[IMG_MAX_MOVE_X].value = (viewWidth / 2) - (imgWidth / 2);
          sharedValues[IMG_MIN_MOVE_X].value = windowWidth - (imgWidth / 2) - (viewWidth / 2);
        } else {
          sharedValues[IMG_MAX_MOVE_X].value = sharedValues[IMG_MIN_MOVE_X].value = (windowWidth - imgWidth) / 2;
        }
        if (viewHeight > windowHeight) {
          sharedValues[IMG_MAX_MOVE_Y].value = (viewHeight / 2) - (imgHeight / 2);
          sharedValues[IMG_MIN_MOVE_Y].value = windowHeight - (imgHeight / 2) - (viewHeight / 2);
        } else {
          sharedValues[IMG_MAX_MOVE_Y].value = sharedValues[IMG_MIN_MOVE_Y].value = (windowHeight - imgHeight) / 2;
        }
      }
      this.calcImgLimit = calcImgLimit
      // 监听上层移动或缩放事件
      wx.worklet.runOnUI(() => {
        'worklet';
        function adjustImg(args) {
          const {
            moveX = null,
            moveY = null,
            scale = null,
            centerX = null,
            centerY = null,
            withAnimation = false
          } = args;
          if (moveX !== null && moveY !== null) {
            if (!moveX && !moveY) return;
            const oldMoveX = sharedValues[MOVE_X].value;
            const oldMoveY = sharedValues[MOVE_Y].value;
            let newMoveX = oldMoveX + moveX;
            let newMoveY = oldMoveY + moveY;
            const minMoveX = sharedValues[IMG_MIN_MOVE_X].value;
            const maxMoveX = sharedValues[IMG_MAX_MOVE_X].value;
            const minMoveY = sharedValues[IMG_MIN_MOVE_Y].value;
            const maxMoveY = sharedValues[IMG_MAX_MOVE_Y].value;
            const isReachXEdge = (newMoveX <= minMoveX || newMoveX >= maxMoveX);
            const isReachYEdge = (newMoveY <= minMoveY || newMoveY >= maxMoveY);
            const isScaleBigger = sharedValues[SCALE].value - sharedValues[INIT_SCALE].value > 0.001;
            const moveGap = Math.abs(moveY) - Math.abs(moveX);
            if (!sharedValues[GESTURE_STATE].value && isReachXEdge && moveGap < 0) {
              // 图片本身不在移动中，或到达水平边缘，或横向移动
              globalThis.eventBus.emit(`${pageId}Toggle`, args);
            } else if (!sharedValues[GESTURE_STATE].value && isReachYEdge && moveGap > 0 && moveY > 0) {
              // 图片本身不在移动中，或到达垂直边缘，或纵向移动
              globalThis.eventBus.emit(`${pageId}Back`, args);
            } else {
              // 移动图片
              const tempOldMoveX = sharedValues[TEMP_MOVE_X].value || oldMoveX;
              const tempOldMoveY = sharedValues[TEMP_MOVE_Y].value || oldMoveY;
              if (isReachXEdge) {
                // 到达水平边缘时增加阻尼
                const offset = tempOldMoveX - oldMoveX;
                const f = 0.52 * ((1 - Math.min(Math.abs(offset) / windowWidth, 1)) ** 2);
                newMoveX = oldMoveX + Math.ceil(moveX * f);
              }
              if (isReachYEdge && isScaleBigger) {
                // 图片放大到达垂直边缘时增加阻尼
                const offset = tempOldMoveY - oldMoveY;
                const f = 0.52 * ((1 - Math.min(Math.abs(offset) / windowHeight, 1)) ** 2);
                newMoveY = oldMoveY + Math.ceil(moveY * f);
              }
              if (sharedValues[GESTURE_STATE].value === 3) {
                // 图片惯性移动中
                const tempNewMoveX = Math.min(Math.max(newMoveX, minMoveX), maxMoveX);
                sharedValues[MOVE_X].value = tempNewMoveX;
                sharedValues[TEMP_MOVE_X].value = tempNewMoveX;
                const tempNewMoveY = Math.min(Math.max(newMoveY, minMoveY), maxMoveY);
                sharedValues[MOVE_Y].value = tempNewMoveY;
                sharedValues[TEMP_MOVE_Y].value = tempNewMoveY;
              } else {
                // 水平方向
                sharedValues[MOVE_X].value = newMoveX;
                const tempNewMoveX = tempOldMoveX + moveX;
                sharedValues[TEMP_MOVE_X].value = Math.min(Math.max(tempNewMoveX, minMoveX), maxMoveX);
                // 垂直方向
                if (isScaleBigger) {
                  // 图片放大情况下设置阻尼
                  sharedValues[MOVE_Y].value = newMoveY;
                  const tempNewMoveY = tempOldMoveY + moveY;
                  sharedValues[TEMP_MOVE_Y].value = Math.min(Math.max(tempNewMoveY, minMoveY), maxMoveY);
                } else {
                  const tempNewMoveY = Math.min(Math.max(newMoveY, minMoveY), maxMoveY);
                  sharedValues[MOVE_Y].value = tempNewMoveY;
                  sharedValues[TEMP_MOVE_Y].value = tempNewMoveY;
                }
                sharedValues[GESTURE_STATE].value = 1;
                globalThis.eventBus.emit(`${pageId}Moving`);
              }
            }
          } else if (scale !== null && centerX !== null && centerY !== null) {
            const oldMoveX = sharedValues[MOVE_X].value;
            const oldMoveY = sharedValues[MOVE_Y].value;
            const oldScale = sharedValues[SCALE].value;
            const newScale = oldScale * scale;
            const tempNewScale = Math.min(Math.max(newScale, sharedValues[IMG_MIN_SCALE].value), sharedValues[IMG_MAX_SCALE].value);
            const realScale = tempNewScale / oldScale;
            // 获得图片放缩时中心偏移变化
            const imgWidth = sharedValues[IMG_WIDTH].value;
            const imgHeight = sharedValues[IMG_HEIGHT].value;
            const gapX = centerX - (imgWidth / 2);
            let newMoveX = gapX - ((gapX - oldMoveX) * realScale);
            const gapY = centerY - (imgHeight / 2);
            let newMoveY = gapY - ((gapY - oldMoveY) * realScale);
            calcImgLimit(tempNewScale, imgWidth, imgHeight);
            newMoveX = Math.min(Math.max(newMoveX, sharedValues[IMG_MIN_MOVE_X].value), sharedValues[IMG_MAX_MOVE_X].value);
            newMoveY = Math.min(Math.max(newMoveY, sharedValues[IMG_MIN_MOVE_Y].value), sharedValues[IMG_MAX_MOVE_Y].value);
            sharedValues[MOVE_X].value = withAnimation ? adjustTiming(newMoveX) : newMoveX;
            sharedValues[MOVE_Y].value = withAnimation ? adjustTiming(newMoveY) : newMoveY;
            if (withAnimation) {
              // 存在动画时无需阻尼
              sharedValues[SCALE].value = adjustTiming(tempNewScale, () => {
                sharedValues[GESTURE_STATE].value = 0;
              })
              sharedValues[TEMP_SCALE].value = tempNewScale;
            } else {
              let calcNewScale = newScale;
              if (Math.abs(newScale - tempNewScale) > 0.001) {
                // 图片超出边界时计算阻尼
                const gap = newScale - oldScale;
                const dampingLimit = gap < 0 ? (newScale / tempNewScale) : (tempNewScale / newScale);
                const f = (gap < 0 ? 0.3 : 0.6) * (Math.min(dampingLimit, 1) ** 2);
                calcNewScale = oldScale + (gap  * f);
              }
              sharedValues[SCALE].value = calcNewScale;
              sharedValues[TEMP_SCALE].value = tempNewScale;
            }
            // 确保不存在回弹
            sharedValues[TEMP_MOVE_X].value = newMoveX;
            sharedValues[TEMP_MOVE_Y].value = newMoveY;
            sharedValues[GESTURE_STATE].value = 2;
            globalThis.eventBus.emit(`${pageId}Moving`);
          }
        }
        // 监听图片拖动手势
        if (!globalThis.temp[`${pageId}${id}ImgMove`]) {
          globalThis.temp[`${pageId}${id}ImgMove`] = args => adjustImg(args);
          globalThis.eventBus.on(`${pageId}${id}Move`, globalThis.temp[`${pageId}${id}ImgMove`]);
        }
        // 监听图片放缩手势
        if (!globalThis.temp[`${pageId}${id}ImgScale`]) {
          globalThis.temp[`${pageId}${id}ImgScale`] = args => adjustImg(args);
          globalThis.eventBus.on(`${pageId}${id}Scale`, globalThis.temp[`${pageId}${id}ImgScale`]);
        }
        // 监听手势结束事件
        if (!globalThis.temp[`${pageId}${id}ImgEnd`]) {
          globalThis.temp[`${pageId}${id}ImgEnd`] = args => {
            // 手势结束
            sharedValues[GESTURE_STATE].value = 0;
            const moveX = sharedValues[MOVE_X].value;
            const targetMoveX = sharedValues[TEMP_MOVE_X].value;
            const needRecoverX = Math.abs(moveX - targetMoveX) > 0.001;
            const moveY = sharedValues[MOVE_Y].value;
            const targetMoveY = sharedValues[TEMP_MOVE_Y].value;
            const needRecoverY = Math.abs(moveY - targetMoveY) > 0.001;
            const scale = sharedValues[SCALE].value;
            const targetScale = sharedValues[TEMP_SCALE].value;
            const needRecoverScale = Math.abs(scale - targetScale) > 0.001;
            const isScaleBigger = sharedValues[SCALE].value - sharedValues[INIT_SCALE].value > 0.001;
            const { velocityX, velocityY } = args;
            const vx = Math.min(Math.abs(velocityX), 700);
            const vy = Math.min(Math.abs(velocityY), 700);
            if (needRecoverX || needRecoverY || needRecoverScale) {
              // 阻尼回弹
              if (needRecoverX) sharedValues[MOVE_X].value = recoverTiming(targetMoveX);
              if (needRecoverY) sharedValues[MOVE_Y].value = recoverTiming(targetMoveY);
              if (needRecoverScale) {
                sharedValues[SCALE].value = recoverTiming(targetScale);
                wx.worklet.runOnJS(wx.vibrateShort('light'))();
              }
            } else if (isScaleBigger && (vx > 50 || vy > 50)) {
              // 惯性计算
              sharedValues[GESTURE_STATE].value = 3;
              const a = -0.0015;
              const directionX = Math.sign(velocityX);
              const directionY = Math.sign(velocityY);
              let vx0 = vx / 1000;
              let vy0 = vy / 1000;
              const t = 16;
              const nextTick = () => {
                'worklet';
                // 进入其他状态时取消惯性
                if (sharedValues[GESTURE_STATE].value !== 3) return;
                let moveX = 0;
                let moveY = 0;
                if (vx0 > 0) {
                  // 若存在水平速度
                  const vx1 = a * t + vx0;
                  if (vx1 > 0) {
                    const s = (vx0 * t) + ((a * (t ** 2)) / 2);
                    moveX = s * directionX;
                    vx0 = vx1;
                  } else {
                    vx0 = 0;
                  }
                }
                if (vy0 > 0) {
                  // 若存在垂直速度
                  const vy1 = a * t + vy0;
                  if (vy1 > 0) {
                    const s = (vy0 * t) + ((a * (t ** 2)) / 2);
                    moveY = s * directionY;
                    vy0 = vy1;
                  } else {
                    vy0 = 0;
                  }
                }
                if (moveX || moveY) {
                  adjustImg({ moveX, moveY });
                  setTimeout(nextTick, t);
                } else {
                  // 回归初始状态
                  sharedValues[GESTURE_STATE].value = 0;
                }
              }
              setTimeout(nextTick, t);
            }
          }
          globalThis.eventBus.on(`${pageId}${id}End`, globalThis.temp[`${pageId}${id}ImgEnd`]);
        }
        // 监听图片双击手势
        if (!globalThis.temp[`${pageId}${id}ImgDoubleTap`]) {
          globalThis.temp[`${pageId}${id}ImgDoubleTap`] = args => adjustImg(args);
          globalThis.eventBus.on(`${pageId}${id}DoubleTap`, globalThis.temp[`${pageId}${id}ImgDoubleTap`]);
        }
      })()
    },
    detached() {
      this.isAttached = false
      // 获得图片唯一标识符
      const id = this.data.image.id
      // 获得页面唯一标识符
      const pageId = this.getPageId()
      // 取消上层移动或缩放监听
      wx.worklet.runOnUI(() => {
        'worklet';
        const removeList = ['Move', 'Scale', 'End', 'DoubleTap'];
        removeList.forEach(item => {
          'worklet';
          const globalKey = `${pageId}${id}Img${item}`;
          if (globalThis.temp[globalKey]) {
            globalThis.eventBus.off(`${pageId}${id}${item}`, globalThis.temp[globalKey]);
            delete globalThis.temp[globalKey];
          }
        })
      })()
    }
  },
  methods: {
    // 重设图片状态
    resetImg() {
      const sharedValues = this.sharedValues ?? [];
      const initMoveX = sharedValues[INIT_MOVE_X].value;
      const initMoveY = sharedValues[INIT_MOVE_Y].value;
      const initScale = sharedValues[INIT_SCALE].value;
      this.calcImgLimit(initScale, sharedValues[IMG_WIDTH].value, sharedValues[IMG_HEIGHT].value);
      sharedValues[MOVE_X].value = initMoveX;
      sharedValues[MOVE_Y].value = initMoveY;
      sharedValues[TEMP_MOVE_X].value = initMoveX;
      sharedValues[TEMP_MOVE_Y].value = initMoveY;
      sharedValues[SCALE].value = initScale;
      sharedValues[TEMP_SCALE].value = initScale;
    },
    // 图片加载完成回调
    onImgLoad(e) {
      const { width: imgWidth, height: imgHeight } = e.detail;
      const windowWidth = getApp().globalData.windowWidth;
      const windowHeight = getApp().globalData.windowHeight;
      const sharedValues = this.sharedValues ?? [];
      const windowRatio = windowWidth / windowHeight;
      sharedValues[IMG_WIDTH].value = imgWidth;
      sharedValues[IMG_HEIGHT].value = imgHeight;
      if (!imgWidth || !imgHeight) return;
      const imgRatio = imgWidth / imgHeight;
      let scale = 1;
      let moveX = 0;
      let moveY = 0;
      if (imgRatio > windowRatio) {
        scale = windowWidth / imgWidth;
        moveX = (windowWidth - imgWidth) / 2;
        const scaleHeight = imgHeight * scale;
        moveY = ((windowHeight - scaleHeight) / 2) - ((imgHeight - scaleHeight) / 2);
        sharedValues[IMG_MAX_SCALE].value = imgWidth > windowWidth ? 2 : DEFAULT_IMG_SCALE_MAX;
        sharedValues[IMG_MIN_SCALE].value = imgWidth > windowWidth ? windowWidth / imgWidth : 1;
      } else {
        scale = windowHeight / imgHeight;
        moveY = (windowHeight - imgHeight) / 2;
        const scaleWidth = imgWidth * scale;
        moveX = ((windowWidth - scaleWidth) / 2) - ((imgWidth - scaleWidth) / 2);
        sharedValues[IMG_MAX_SCALE].value = imgHeight > windowHeight ? 2 : DEFAULT_IMG_SCALE_MAX;
        sharedValues[IMG_MIN_SCALE].value = imgHeight > windowHeight ? windowHeight / imgHeight : 1;
      }
      this.calcImgLimit(scale, imgWidth, imgHeight);
      sharedValues[MOVE_X].value = moveX;
      sharedValues[MOVE_Y].value = moveY;
      sharedValues[TEMP_MOVE_X].value = moveX;
      sharedValues[TEMP_MOVE_Y].value = moveY;
      sharedValues[SCALE].value = scale;
      sharedValues[TEMP_SCALE].value = scale;
      sharedValues[INIT_MOVE_X].value = moveX;
      sharedValues[INIT_MOVE_Y].value = moveY;
      sharedValues[INIT_SCALE].value = scale;
      this.isImgLoaded = true;
      if (this.data.status === 1) {
        this.triggerEvent('render', this.data.image);
      }
    },
    // 双击事件处理
    onDoubleTap(e) {
      'worklet';
      const sharedValues = this.sharedValues ?? [];
      const currentScale = sharedValues[SCALE].value;
      const minScale = sharedValues[IMG_MIN_SCALE].value;
      globalThis.eventBus.emit(`${sharedValues[PAGE_ID].value}${sharedValues[CURRENT_ID].value}DoubleTap`, {
        scale: currentScale <= minScale ? 2 : 0.0001,
        centerX: e.absoluteX,
        centerY: e.absoluteY,
        withAnimation: true
      });
    }
  }
})