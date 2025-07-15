App({
  onLaunch() {
    this.calcPageTopPaddingAndNaigationBarInfo(),
    this.getWindowHeightAndWidthInfo(),
    this.calcSafeBottom(),
    this.calcTarbarHeight(),
    this.calcDisplayAreaHeight(),
    this.loadStorageLoginInfo()
  },
  
  globalData: {
    baseUrl: 'http://127.0.0.1:8081/',
    navigationBarHeight: 0,
    windowHeight: 0,
    windowWidth: 0,
    safeAreaBottomHeight: 0,
    tarbarHeight: 0,
    displayAreaHeight: 0,
    loginInfo: {
      loginStatus: 'false',
      loginAccount: '',
      loginPassword: '',
      loginAvatar: '',
      loginNickname: ''
    }
  },

  // 计算页面填充高度和导航栏高度
  calcPageTopPaddingAndNaigationBarInfo() {
    const systemInfo = wx.getWindowInfo();
    const menuButtonInfo = wx.getMenuButtonBoundingClientRect();
    // 导航栏高度 = 页面填充高度 = 状态栏到胶囊的间距（胶囊上坐标位置-状态栏高度） * 2 + 胶囊高度 + 状态栏高度
    this.globalData.navigationBarHeight = (menuButtonInfo.top - systemInfo.statusBarHeight) * 2 + menuButtonInfo.height + systemInfo.statusBarHeight;
  },

  // 获得页面高度和宽度
  getWindowHeightAndWidthInfo() {
    const systemInfo = wx.getWindowInfo();
    this.globalData.windowHeight = systemInfo.windowHeight;
    this.globalData.windowWidth = systemInfo.windowWidth;
  },

  // 计算页面底部安全区高度
  calcSafeBottom() {
    this.safeAreaBottomHeight = 34;
    if (wx.getWindowInfo().safeArea) {
      const windowHeight = wx.getWindowInfo().windowHeight;
      this.globalData.safeAreaBottomHeight = windowHeight - wx.getWindowInfo().safeArea.bottom;
    }
  },

  // 计算底部导航栏高度
  calcTarbarHeight() {
    const windowWidth = wx.getWindowInfo().windowWidth;
    this.globalData.tarbarHeight = this.safeAreaBottomHeight + 95 * windowWidth / 750;
  },

  // 计算页面内容部分高度
  calcDisplayAreaHeight() {
    const windowHeight = wx.getWindowInfo().windowHeight;
    // 页面内容显示高度 = 页面高度 - 导航栏高度 - tarbar高度
    this.globalData.displayAreaHeight = windowHeight - this.globalData.navigationBarHeight - this.globalData.tarbarHeight;
  },

  // 加载存储在本地的用户登录信息
  loadStorageLoginInfo() {
    // 如果存储的用户信息存在
    if (wx.getStorageSync('loginStatus', 'false') === 'true') {
      this.globalData.loginInfo.loginStatus = wx.getStorageSync('loginStatus', 'false');
      this.globalData.loginInfo.loginAccount = wx.getStorageSync('loginAccount', '');
      this.globalData.loginInfo.loginPassword = wx.getStorageSync('loginPassword', '');
      this.globalData.loginInfo.loginAvatar = wx.getStorageSync('loginAvatar', '');
      this.globalData.loginInfo.loginNickname = wx.getStorageSync('loginNickname', '');
    }
  }
});