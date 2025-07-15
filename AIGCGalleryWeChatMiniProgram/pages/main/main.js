Page({
  data: {
    topMarginHeight: getApp().globalData.navigationBarHeight,
    bottomMarginHeight: getApp().globalData.tarbarHeight,
    currentView: "roam",
    account: getApp().globalData.loginInfo.loginAccount
  },
  onShow() {
    // 页面显示时更新导航栏内用户信息显示
    this.selectComponent('#main-navigation').updateIndividualInfo();
    // 账号发生变化时触发页面刷新
    this.refreshMainPage();
  },
  // 处理页面选择与跳转事件
  handleChooseAndJumpEvent(evt) {
    const { navId } = evt.detail;
    if (getApp().globalData.loginInfo.loginStatus === 'false' && navId !== 'roam') {
      // 若前往创作和关注圈时未登录
      wx.navigateTo({ url: '/pages/login/login' });
      return;
    }
    this.setData({ currentView: navId });
  },
  // 账号发生变化时触发页面刷新
  refreshMainPage() {
    if (getApp().globalData.loginInfo.loginAccount !== this.data.account) {
      this.setData({ 'account': getApp().globalData.loginInfo.loginAccount });
      this.selectComponent('#main-tarbar').manualHandleFunctionChooseAndJump("roam");
    }
  }
});