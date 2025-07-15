Page({
  data: {
    windowHeight: getApp().globalData.windowHeight,
    navigationBarHeight: getApp().globalData.navigationBarHeight,
    // 设置顶部覆盖区是否可见
    showTopCover: false,
    // 用户基本信息
    loginData: {
      avatar: '/images/default_avatar.png',
      nickname: ''
    },
    // 用户粉丝与关注
    follow: {
      followedNum: 0,
      followerNum: 0
    },
    // 设置个人设置菜单是否可见
    showIndividualSettings: false,
    // 个人设置菜单选项
    individualSettingsItem: [
      { text: '更改头像', value: 1 },
      { text: '改变昵称', value: 2 },
      { text: '修改密码', value: 3 },
      { text: '退出登录', type: 'warn', value: 4 }
    ],
    // 个人作品显示区域最小高度
    individualMomentDisplayAreaHeight: 0,
    momentList: [],
    loadInfo: {
      // 当前页
      currentPage: 1,
      // 单页条数
      pageSize: 6,
      // 是否还有更多数据
      moreArt: true,
      // 是否在加载
      isLoading: false
    },
    showFirstLoading: false,
    showFailLoad: false,
    nextLoadTip: '',
    currentYear: new Date().getFullYear(),
    convRate: 0.5
  },
  onReady() {
    // 请求关注与粉丝数据
    this.loadFollowCountData()
    // 首次加载个人作品数据
    this.openLoadIndividualMoment()
    // 计算个人作品显示区域最小高度和像素转换比例
    this.setData({
      'convRate': getApp().globalData.windowWidth / 750,
      'individualMomentDisplayAreaHeight': getApp().globalData.windowHeight - getApp().globalData.windowWidth * 0.8
    });
  },
  onShow() {
    // 加载用户基本信息
    this.setData({
      'loginData.avatar': getApp().globalData.loginInfo.loginAvatar,
      'loginData.nickname': getApp().globalData.loginInfo.loginNickname
    });
  },
  // 显示个人设置菜单
  showIndividualSettingsMenu() {
    this.setData({ 'showIndividualSettings': true });
  },
  // 个人设置菜单点击响应
  individualSettingsClick(e) {
    let { value } = e.detail;
    if (value === 1) {
      // 更改头像
      this.modifyAvatar();
    } else if (value === 2) {
      // 改变昵称
      wx.navigateTo({
        url: '/pages/modify/modify?modifyType=nickname'
      });
    } else if (value === 3) {
      // 修改密码
      wx.navigateTo({
        url: '/pages/modify/modify?modifyType=password'
      });
    } else {
      // 退出登录
      this.logout();
    }
    this.setData({ 'showIndividualSettings': false });
  },
  // 修改头像
  modifyAvatar() {
    // 头像修改震动反馈
    wx.vibrateShort('light');
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['album'],
      sizeType: ['compressed'],
      success: (res) => {
        // 格式检测
        if (!/(\.jpg|\.png|\.jpeg)$/.test(res.tempFiles[0].tempFilePath.toLowerCase())) {
          wx.showToast({
            title: '请上传jpg、png或jpeg格式的照片',
            icon: 'none',
            duration: 2500
          });
          return;
        }
        // 将图片裁剪为正方形
        wx.cropImage({
          src: res.tempFiles[0].tempFilePath,
          cropScale: '1:1',
          success: (res) => {
            // 压缩图片
            wx.compressImage({
							src: res.tempFilePath,
							quality: 80,
							success: (res) => {
                // 将头像图片转为base64
                wx.getFileSystemManager().readFile({
                  filePath: res.tempFilePath,
                  encoding: 'base64',
                  success: (res) => {
                    const avatar = 'data:image/png;base64,' + res.data;
                    // 提交头像修改
                    wx.request({
                      method: 'POST',
                      url: getApp().globalData.baseUrl + 'identity/modifyAvatar',
                      header: {
                        'Content-Type': 'application/x-www-form-urlencoded'
                      },
                      data: {
                        account: getApp().globalData.loginInfo.loginAccount,
                        password: getApp().globalData.loginInfo.loginPassword,
                        avatar: avatar
                      },
                      success: (res) => {
                        if (res.data.success === true) {
                          // 头像修改成功
                          wx.setStorageSync('loginAvatar', avatar);
                          getApp().globalData.loginInfo.loginAvatar = avatar;
                          this.setData({ 'loginData.avatar': avatar });
                          // 头像修改成功提示
                          wx.showToast({
                            icon: 'none',
                            title: '头像修改成功',
                            duration: 1500
                          });
                        } else {
                          // 头像修改失败
                          let title = '';
                          if (res.data.result === '账号不存在') {
                            // 提示账号不存在
                            title = '账号不存在';
                          } else if (res.data.result === '密码错误') {
                            // 提示身份验证失败
                            title = '身份验证失败';
                          } else {
                            // 出现头像修改异常
                            title = '头像修改异常';
                          }
                          wx.showToast({
                            icon: 'error',
                            title: title,
                            duration: 2500
                          });
                        }
                      },
                      fail: () => {
                        wx.showToast({
                          icon: 'error',
                          title: '网络异常',
                          duration: 2500
                        });
                      }
                    });
                  },
                  fail: () => {
                    wx.showToast({
                      icon: 'error',
                      title: '图片转码失败',
                      duration: 2500
                    });
                    return;
                  }
                });
              },
              fail: () => {
                wx.showToast({
                  icon: 'error',
                  title: '图片压缩失败',
                  duration: 2500
                });
              }
            });
          },
          fail: () => {
            wx.showToast({
              icon: 'error',
              title: '图片裁剪失败',
              duration: 2500
            });
          }
        });
      },
      fail: () => {
        wx.showToast({
          icon: 'error',
          title: '图片选取失败',
          duration: 2500
        });
      }
    });
  },
  // 退出登录
  logout() {
    wx.removeStorageSync('loginStatus');
    wx.removeStorageSync('loginAccount');
    wx.removeStorageSync('loginPassword');
    wx.removeStorageSync('loginAvatar');
    wx.removeStorageSync('loginNickname');
    getApp().globalData.loginInfo.loginStatus = 'false';
    getApp().globalData.loginInfo.loginAccount = '';
    getApp().globalData.loginInfo.loginPassword = '';
    getApp().globalData.loginInfo.loginAvatar = '';
    getApp().globalData.loginInfo.loginNickname = '';
    // 返回主页
    wx.navigateBack();
    // 退出登录提示
    wx.showToast({
      icon: 'none',
      title: '已退出登录',
      duration: 1500
    });
  },
  // 请求关注与粉丝数据
  loadFollowCountData() {
    wx.request({
      method: 'POST',
      url: getApp().globalData.baseUrl + 'follow/getFollowCount',
      header: {
        'Content-Type': 'application/x-www-form-urlencoded'
      },
      data: { account: getApp().globalData.loginInfo.loginAccount },
      success: (res) => {
        // 判断粉丝与关注数据是否有效
        if (res.data.followedNum === -1 || res.data.followerNum === -1) {
          wx.showToast({
            icon: 'error',
            title: '数据异常',
            duration: 2500
          });
        } else {
          this.setData({
            'follow.followedNum': res.data.followedNum,
            'follow.followerNum': res.data.followerNum
          });
        }
      },  
      fail: () => {
        wx.showToast({
          icon: 'error',
          title: '数据请求失败',
          duration: 2500
        });
      }
    });
  },
  // 首次加载个人作品数据
  openLoadIndividualMoment() {
    this.setData({
      'loadInfo.isLoading': true,
      'showFirstLoading': true,
      'showFailLoad': false
    });
    wx.request({
      method: 'POST',
      url: getApp().globalData.baseUrl + 'post/getIndividualMoments',
      header: {
        'Content-Type': 'application/x-www-form-urlencoded'
      },
      data: {
        account: getApp().globalData.loginInfo.loginAccount,
        page: this.data.loadInfo.currentPage,
        size: this.data.loadInfo.pageSize
      },
      success: (res) => {
        this.setData({
          'momentList': res.data.momentList,
          'loadInfo.moreArt': res.data.hasNextPage,
          'showFirstLoading': false
        });
        if (!this.data.loadInfo.moreArt) {
          this.setData({ 'nextLoadTip': '无更多数据，请发表您的第一个作品吧' });
        }
      },
      fail: () => {
        this.setData({
          'showFirstLoading': false,
          'showFailLoad': true
        });
      },
      complete: () => {
        this.setData({ 'loadInfo.isLoading': false });
      }
    });
  },
  // 加载更多关注圈数据
  tryLoadMoreIndividualMoment() {
    this.setData({ 'nextLoadTip': '加载中' });
    wx.request({
      method: 'POST',
      url: getApp().globalData.baseUrl + 'post/getIndividualMoments',
      header: {
        'Content-Type': 'application/x-www-form-urlencoded'
      },
      data: {
        account: getApp().globalData.loginInfo.loginAccount,
        page: this.data.loadInfo.currentPage,
        size: this.data.loadInfo.pageSize
      },
      success: (res) => {
        this.setData({
          'momentList': this.data.momentList.concat(res.data.momentList),
          'loadInfo.moreArt': res.data.hasNextPage
        });
        if (this.data.loadInfo.moreArt) {
          this.setData({ 'nextLoadTip': '' });
        } else {
          this.setData({ 'nextLoadTip': '无更多数据，请发表更多作品吧' });
        }
      },
      fail: () => {
        this.setData({ 'nextLoadTip': '加载数据异常，请上滑重试' });
      }
    });
  },
  // 页面触底加载更多关注圈数据
  onReachBottom() {
    this.isMoreIndividualMomentToLoad();
  },
  // 页面向下滚动显示顶部覆盖区
  onShowTopCover(e) {
    const scrollTop = e.detail.scrollTop;
    if (scrollTop > this.data.convRate * 260) {
      this.setData({ 'showTopCover': true });
    } else {
      this.setData({ 'showTopCover': false });
    }
  },
  // 判断是否继续加载数据
  isMoreIndividualMomentToLoad() {
    if (this.data.loadInfo.moreArt && !this.data.loadInfo.isLoading) {
      this.setData({ 'loadInfo.isLoading': true });
      this.setData({ 'loadInfo.currentPage': Number(++this.data.loadInfo.currentPage) });
      this.tryLoadMoreIndividualMoment();
      this.setData({ 'loadInfo.isLoading': false });
    }
  },
  // 触发图片预览
  onTapImage(e) {
    const image = e.currentTarget.dataset.image;
    // 预览图像
    wx.previewImage({
      urls: [image],
      showmenu: false,
      fail: () => {
        wx.showToast({
          icon: 'error',
          title: '图像预览失败',
          duration: 2500
        });
      }
    });
  },
  // 触发视频预览
  onTapVideo(e) {
    const cover = e.currentTarget.dataset.cover;
    const video = e.currentTarget.dataset.video;
    // 预览视频
    wx.previewMedia({
      sources: {
        url: video,
        type: 'video',
        poster: cover
      },
      showmenu: false,
      fail: () => {
        wx.showToast({
          icon: 'error',
          title: '视频预览失败',
          duration: 2500
        });
      }
    });
  }
})