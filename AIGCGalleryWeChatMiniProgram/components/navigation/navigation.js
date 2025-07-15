Component({
	data: {
    navigationBarHeight: getApp().globalData.navigationBarHeight,
    menuButtonTopPaddingHeight: wx.getMenuButtonBoundingClientRect().top,
    menuButtonHeight: wx.getMenuButtonBoundingClientRect().height,
    menuButtonRightPaddingWidth: 0,
    navigationContentMaxWidth: 0,
    loginInfo: {
      // 登录状态
      loginStatus: '',
      // 登录中的用户头像
      loginAvatar: '',
      // 登录中的用户昵称
      loginNickname: ''
    }
  },
  lifetimes: {
    created() {
      // 计算导航栏内容区左侧填充量与最大长度
      this.setData({
        'menuButtonRightPaddingWidth': getApp().globalData.windowWidth - wx.getMenuButtonBoundingClientRect().right,
        'navigationContentMaxWidth': wx.getMenuButtonBoundingClientRect().left - 2 * getApp().globalData.windowWidth + 2 * wx.getMenuButtonBoundingClientRect().right
      });
    },
    attached() {
      // 更新用户登录信息
      this.updateIndividualInfo();
    }
  },
  methods: {
    // 更新用户登录信息
    updateIndividualInfo() {
      this.setData({
        'loginInfo.loginStatus': getApp().globalData.loginInfo.loginStatus,
        'loginInfo.loginAvatar': getApp().globalData.loginInfo.loginAvatar,
        'loginInfo.loginNickname': getApp().globalData.loginInfo.loginNickname
      });
    },
    // 跳转至个人主页或登录页
    jumpToLoginOrIndividual() {
      let targetUrl = '';
      if (this.data.loginInfo.loginStatus === 'true') {
        targetUrl = '/pages/individual/individual';
      } else if (this.data.loginInfo.loginStatus === 'false') {
        targetUrl = '/pages/login/login';
      }
      // 个人主页或登录页跳转震动反馈
      wx.vibrateShort('light');
      wx.navigateTo({
        url: targetUrl
      });
    }
  }
})