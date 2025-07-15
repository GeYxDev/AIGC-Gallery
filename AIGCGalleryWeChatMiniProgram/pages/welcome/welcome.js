// 获得窗口高度和导航栏高度
const { windowHeight, navigationBarHeight } = getApp().globalData

Page({
  data: {
    topMarginHeight: navigationBarHeight
  },
  onLoad() {
    wx.router.addRouteBuilder('welcomeToMainRoute', welcomeToMainRouteBuilder);
    const translateY = wx.worklet.shared(0);
    const opacity = wx.worklet.shared(1);
    this.applyAnimatedStyle('#tipAnimation', () => {
      'worklet';
      return {
        transform: `translateY(${translateY.value}px)`,
        opacity: opacity.value
      };
    });
    this.slideToBeginTransformAnimation(translateY);
    this.slideToBeginOpacityAnimation(opacity);
    this.gestureTransY = wx.worklet.shared(0);
    this.isSlideTriggerNavigateTo = wx.worklet.shared(false);
    this.applyAnimatedStyle('#slideToBeginGesture', () => {
      'worklet';
      return {
        transform: `translateY(${this.gestureTransY.value}px)`,
      };
    });
  },
  // 滑动开始提示非线性动画：移动
  slideToBeginTransformAnimation(translateY) {
    const moveUpAnimation = wx.worklet.timing(-25, {
      duration: 1500,
      easing: wx.worklet.Easing.inOut(wx.worklet.Easing.sin)
    });
    const moveDownAnimation = wx.worklet.timing(0, {
      duration: 2500,
      easing: wx.worklet.Easing.inOut(wx.worklet.Easing.sin)
    });
    const moveHoldAnimation = wx.worklet.timing(0, {
      duration: 100,
      easing: wx.worklet.Easing.inOut(wx.worklet.Easing.sin)
    });
    const sequenceAnimation = wx.worklet.sequence(moveUpAnimation, moveDownAnimation, moveHoldAnimation);
    const loopAnimation = wx.worklet.repeat(sequenceAnimation, -1);
    translateY.value = loopAnimation;
  },
  // 滑动开始提示非线性动画：移动
  slideToBeginOpacityAnimation(opacity) {
    const fadeInAnimation = wx.worklet.timing(0.3, {
      duration: 1500,
      easing: wx.worklet.Easing.inOut(wx.worklet.Easing.sin)
    });
    const fadeOutAnimation = wx.worklet.timing(1.0, {
      duration: 2500,
      easing: wx.worklet.Easing.inOut(wx.worklet.Easing.sin)
    });
    const fadeHoldAnimation = wx.worklet.timing(1.0, {
      duration: 100,
      easing: wx.worklet.Easing.inOut(wx.worklet.Easing.sin)
    });
    const sequenceAnimation = wx.worklet.sequence(fadeInAnimation, fadeOutAnimation, fadeHoldAnimation);
  const loopAnimation = wx.worklet.repeat(sequenceAnimation, -1);
    opacity.value = loopAnimation;
  },
  // 页面跳转
  jumpToMain() {
    wx.navigateTo({
      url: '/pages/main/main',
      routeType: 'welcomeToMainRoute'
    })
  },
  // 更新拖动动画并判定是否跳转页面
  handleDragUpdate(delta) {
    'worklet';
    const gestureTransY = this.gestureTransY.value + delta;
    this.gestureTransY.value = gestureTransY;
  },
  // 结束拖动时末端处理与页面跳转判定
  handleDragEnd(velocity) {
    'worklet';
    const speed = velocity / windowHeight;
    if (Math.abs(speed) >= 0.6 && !this.isSlideTriggerNavigateTo.value) {
      this.isSlideTriggerNavigateTo.value = true;
      transitionDuration.value = 1000 / Math.abs(speed);
      wx.worklet.runOnJS(this.jumpToMain)();
    }
    if (!this.isSlideTriggerNavigateTo.value) {
      // 未触发路由时动画处理
      this.gestureTransY.value = wx.worklet.spring(
        0.0, {
          mass: 5,
          overshootClamping: true,
          velocity: velocity
        },
        () => { 'worklet'; }
      );
    } else {
      // 触发路由时动画处理
      this.gestureTransY.value = wx.worklet.spring(
        0.0, {
          mass: 10,
          overshootClamping: true,
          velocity: velocity
        },
        () => { 'worklet'; }
      );
      // 触发路由时震动反馈
      wx.vibrateShort('light');
    }
    this.isSlideTriggerNavigateTo.value = false;
  },
  // 滑动手势处理
  slideToBeginGesture(gestureEvt) {
    'worklet';
    if (gestureEvt.state === 2) {
      this.handleDragUpdate(gestureEvt.deltaY);
    } else if (gestureEvt.state === 3) {
      this.handleDragEnd(gestureEvt.velocityY);
    } else if (gestureEvt.state === 4) {
      this.handleDragEnd(0.0);
    }
  }
});

// 路由向前跳转动画持续时间
const transitionDuration = wx.worklet.shared(1000);

// 欢迎页面跳转至主页面动画
const welcomeToMainRouteBuilder = (CustomRouteContext) => {
  const {
    primaryAnimation,
    userGestureInProgress
  } = CustomRouteContext
  const handlePrimaryAnimation = () => {
    'worklet'
    let t = primaryAnimation.value
    if (!userGestureInProgress.value) {
      t = wx.worklet.Easing.bezier(0.35, 0.91, 0.33, 0.97).factory()(t)
    }
    const transY = windowHeight * (1 - t)
    return {
      transform: `translateY(${transY}px)`
    }
  }
  const handlePreviousPageAnimation = () => {
		'worklet'
		let t = primaryAnimation.value
		if (!userGestureInProgress.value) {
			t = wx.worklet.Easing.bezier(0.35, 0.91, 0.33, 0.97).factory()(t)
    }
    const transY = -windowHeight * t
    return {
      transform: `translateY(${transY}px)`
    }
	}
  return {
    opaque: false,
    handlePrimaryAnimation,
    handlePreviousPageAnimation,
    transitionDuration: transitionDuration.value,
		reverseTransitionDuration: 1200,
		canTransitionTo: false,
		canTransitionFrom: true
  }
}