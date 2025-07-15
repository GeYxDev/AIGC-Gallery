Component({
  properties: {
    navId: {
      type: String,
      value: "roam"
    }
  },
  data: {
    tarbarSafeHeight: getApp().globalData.safeAreaBottomHeight,
    tarbarTotalHeight: getApp().globalData.tarbarHeight
  },
  methods: {
    // 处理按钮与页面选择
    functionChooseAndJumpEvent(evt) {
      // 按钮与页面选择震动反馈
      wx.vibrateShort('light');
      const navId = evt.currentTarget.dataset.id;
      this.setData({ navId: navId });
      this.triggerEvent('nav', { navId: navId });
    },
    // 手动处理按钮与页面选择
    manualHandleFunctionChooseAndJump(navId) {
      this.setData({ navId: navId });
      this.triggerEvent('nav', { navId: navId });
    }
  }
});