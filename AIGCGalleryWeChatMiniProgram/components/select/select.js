Component({
  properties: {
    // 项目标题
    title: {
      type: String,
      value: ''
    },
    // 选项列表
    optionList: {
      type: Array,
      value: []
    }
  },
  data: {
    // 当前选项索引
    currentIndex: 0,
    // 当前选择项
    currentOption: ''
  },
  lifetimes: {
    attached() {
      // 初始选择项
      const initalOption = this.data.optionList[this.data.currentIndex];
      // 赋值初始选择项
      this.setData({ 'currentOption': initalOption });
      // 传递初始选择项
      this.triggerEvent("handleOptionChange", { option: initalOption });
    }
  },
  methods: {
    // 点击切换选项后显示并传递选择
    handleChangeOptionOperation() {
      // 选项切换震动反馈
      wx.vibrateShort('light');
      let newIndex = this.data.currentIndex;
      let newOption = this.data.currentOption;
      if (this.data.currentIndex >= this.data.optionList.length - 1) {
        newIndex = 0;
        newOption = this.data.optionList[0];
      } else {
        newIndex += 1;
        newOption = this.data.optionList[newIndex];
      }
      this.setData({
        'currentIndex': newIndex,
        'currentOption': newOption
      });
      this.triggerEvent("handleOptionChange", { option: newOption });
    }
  }
})