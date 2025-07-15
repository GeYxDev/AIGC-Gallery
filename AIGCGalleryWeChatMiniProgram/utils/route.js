// 动画状态
export const AnimationStatus = {
  dismissed: 0, // The animation is stopped at the beginning.
  forward: 1, // The animation is running from beginning to end.
  reverse: 2, // The animation is running backwards, from end to beginning.
  completed: 3, // The animation is stopped at the end.
}

// 引入函数
const { Easing, shared, derived } = wx.worklet

// 动画曲线效果
export const Curves = {
  linearToEaseOut: Easing.cubicBezier(0.35, 0.91, 0.33, 0.97),
  easeInToLinear: Easing.cubicBezier(0.67, 0.03, 0.65, 0.09),
  fastOutSlowIn: Easing.cubicBezier(0.4, 0.0, 0.2, 1.0),
  slowOutFastIn: Easing.cubicBezier(0.0, 0.8, 1.0, 0.6),
  easeOutCubic: Easing.cubicBezier(0.215, 0.61, 0.355, 1.0),
  easeInExpo: Easing.cubicBezier(0.7, 0, 0.84, 0),
}

// 动画曲线定义
export function CurveAnimation({ animation, animationStatus, curve, reverseCurve }) {
  return derived(() => {
    'worklet'
    const useForwardCurve = !reverseCurve || animationStatus.value !== AnimationStatus.reverse;
    const activeCurve = useForwardCurve ? curve : reverseCurve;
    const t = animation.value;
    if (!activeCurve) { return t };
    if (t === 0 || t === 1) { return t };
    return activeCurve(t);
  })
}

// 线性插值
export const lerp = (begin, end, t) => {
  'worklet'
  return begin + (end - begin) * t
}

// 数值截断
export const clamp = function (cur, lowerBound, upperBound) {
  'worklet';
  if (cur > upperBound) return upperBound;
  if (cur < lowerBound) return lowerBound;
  return cur;
};

// 手势状态
export const GestureState = {
  POSSIBLE: 0, // 0 此时手势未识别，如 panDown等
  BEGIN: 1, // 1 手势已识别
  ACTIVE: 2, // 2 连续手势活跃状态
  END: 3, // 3 手势终止
  CANCELLED: 4 // 4 手势取消
}

// 浏览手势状态
export const PreviewerGesture = {
  Init: 0, // 初始
  Moving: 1, // 移动图片
  Toggle: 2, // 切换图片
  Back: 3, // 退出页面
}

// 动画飞行状态
export const FlightDirection = {
  PUSH: 0,
  POP: 1,
}

// 操作恢复动画时间
export function recoverTiming(target, callback) {
  'worklet'
  return wx.worklet.timing(target, { duration: 200 }, callback)
}

// 操作持续动画时间
export function adjustTiming(target, callback) {
  'worklet'
  return wx.worklet.timing(target, { duration: 300 }, callback)
}

// 计算透明度
export function calcOpacity(moveY, screenHeight) {
  'worklet'
  const opacityRatio = moveY / (screenHeight / 2)
  return clamp((1 - opacityRatio) ** 3, 0, 1)
}

// 计算缩放
export function calcScale(moveY, screenHeight) {
  'worklet'
  const scaleRange = 0.4
  const scaleRatio = moveY / (screenHeight / 3 * 2)
  return clamp(1 - scaleRange * scaleRatio, 0.6, 1)
}

// detail <-> card路由动画
const scaleTransitionRouteBuilder = (routeContext) => {
  const {
    primaryAnimation,
    primaryAnimationStatus,
    userGestureInProgress,
  } = routeContext;
  const shareEleTop = shared(0);
  routeContext.shareEleTop = shareEleTop;
  const _curvePrimaryAnimation = CurveAnimation({
    animation: primaryAnimation,
    animationStatus: primaryAnimationStatus,
    curve: Easing.in(Curves.fastOutSlowIn),
    reverseCurve: Easing.out(Curves.fastOutSlowIn),
  });
  // 每次路由动画后清理变量
  const reset = () => {
    'worklet'
    if (globalThis['RouteCardSrcRect']) {
      globalThis['RouteCardSrcRect'].value = undefined;
    }
    if (globalThis['RouteCardDestRect']) {
      globalThis['RouteCardDestRect'].value = undefined;
    }
  }
  const handlePrimaryAnimation = () => {
    'worklet'
    const status = primaryAnimationStatus.value;
    // 手势返回时动画在详情页处理
    if (userGestureInProgress.value) {
      return {
        opacity: Easing.out(Easing.cubicBezier(0.5, 0, 0.7, 0.5)(primaryAnimation.value)),
      }
    }
    if (status == AnimationStatus.dismissed) {
      reset();
      return {
        transform: `translate(0, 0) scale(0)`,
      }
    }
    if (status == AnimationStatus.completed) {
      reset();
      return {
        transform: `translate(0, 0) scale(1)`,
      }
    }
    let transX = 0;
    let transY = 0;
    let scale = status === AnimationStatus.reverse ? 1 : 0;
    // 进入或者接口返回时赋值
    if (globalThis['RouteCardSrcRect'] && globalThis['RouteCardSrcRect'].value != undefined) {
      const begin = globalThis['RouteCardSrcRect'].value;
      const end = globalThis['RouteCardDestRect'].value;
      if (status === AnimationStatus.forward) {
        shareEleTop.value = end.top;
      }
      let t = _curvePrimaryAnimation.value;
      if (status === AnimationStatus.reverse || status === AnimationStatus.dismissed) {
        t = 1 - t;
      }
      const shareEleX = lerp(begin.left, end.left, t);
      const shareEleY = lerp(begin.top, end.top, t);
      const shareEleW = lerp(begin.width, end.width, t);
      transX = shareEleX;
      if (status === AnimationStatus.reverse) {
        scale = shareEleW / begin.width;
        transY = shareEleY - begin.top * scale;
      } else {
        scale = shareEleW / end.width;
        transY = shareEleY - end.top * scale;
      }
    }
    return {
      transform: `translate(${transX}px, ${transY}px) scale(${scale})`,
      transformOrigin: '0 0',
      opacity: _curvePrimaryAnimation.value,
    }
  }
  return {
    opaque: false,
    handlePrimaryAnimation,
    transitionDuration: 250,
    reverseTransitionDuration: 250,
    canTransitionTo: false,
    canTransitionFrom: false,
    barrierColor: "rgba(0, 0, 0, 0.3)",
  }
}

// 判断是否已返回过RouteBuilder
let hasInstalled = false
export function installScaleTransitionRouteBuilder() {
  if (hasInstalled) {
    return;
  }
  wx.router.addRouteBuilder('cardScaleTransition', scaleTransitionRouteBuilder);
  hasInstalled = true;
}

// 预览媒体至浏览界面路由动画RouteBuilder
export function installPointScaleToggleRouteBuilder() {
  wx.router.addRouteBuilder('pointScaleToggle', ({ primaryAnimation }) => {
    const handlePrimaryAnimation = () => {
      'worklet'
      return {
        opacity: Curves.fastOutSlowIn(primaryAnimation.value),
      }
    }
    return {
      opaque: false,
      handlePrimaryAnimation,
      transitionDuration: 300,
      reverseTransitionDuration: 300,
      canTransitionTo: false,
      canTransitionFrom: false,
    }
  })
}