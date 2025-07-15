// 引入MD5加密
import { hexMD5 } from '../../utils/md5'

Page({
  data: {
    topMarginHeight: getApp().globalData.navigationBarHeight,
    loginData: {
      account: '',
      password: ''
    },
    errors: {
      account: '',
      password: ''
    },
    passwordVisibility: false
  },
  // 改变密码可见性
  changeVisibility() {
    this.setData({ 'passwordVisibility': !this.data.passwordVisibility });
  },
  // 账号输入处理
  accountInputChange(e) {
    this.setData({ 'loginData.account': e.detail.value });
    this.verifyAccount(e.detail.value);
  },
  // 账号正确性验证
  verifyAccount(value) {
    let error = '';
    if (!value) {
      error = '账号不能为空';
    }
    this.setData({ 'errors.account': error });
  },
  // 密码输入处理
  passwordInputChange(e) {
    this.setData({ 'loginData.password': e.detail.value });
    this.verifyPassword(e.detail.value);
  },
  // 密码正确性验证
  verifyPassword(value) {
    let error = '';
    if (!value) {
      error = '密码不能为空';
    }
    this.setData({ 'errors.password': error });
  },
  // 跳转至注册界面
  jumpToRegistration() {
    wx.navigateTo({
      url: '/pages/register/register'
    });
  },
  // 提交登录操作
  commitLoginOperation() {
    let errors = {};
    if (!this.data.loginData.account) {
        errors.account = '账号不能为空';
    }
    if (!this.data.loginData.password) {
        errors.password = '密码不能为空';
    }
    this.setData({ 'errors': errors });
    // 不存在错误提示时可以提交登录
    if (!errors.account && !errors.password) {
      // 登录提交操作动态反馈
      wx.vibrateShort('light');
      // md5加密密码
      const encodedPassword = hexMD5(this.data.loginData.password);
      // 向服务器请求校验
      wx.request({
        method: 'POST',
        url: getApp().globalData.baseUrl + 'identity/login',
        header: {
          'Content-Type': 'application/x-www-form-urlencoded'
        },
        data: {
          account: this.data.loginData.account,
          password: encodedPassword
        },
        success: (res) => {
          if (res.data.success === true) {
            // 登录成功
            wx.setStorageSync('loginStatus', 'true');
            wx.setStorageSync('loginAccount', res.data.result.account);
            wx.setStorageSync('loginPassword', encodedPassword);
            wx.setStorageSync('loginAvatar', res.data.result.avatar);
            wx.setStorageSync('loginNickname', res.data.result.nickname);
            getApp().globalData.loginInfo.loginStatus = 'true';
            getApp().globalData.loginInfo.loginAccount = res.data.result.account;
            getApp().globalData.loginInfo.loginPassword = encodedPassword;
            getApp().globalData.loginInfo.loginAvatar = res.data.result.avatar;
            getApp().globalData.loginInfo.loginNickname = res.data.result.nickname;
            // 返回原页面
            wx.navigateBack();
            // 登录成功提示
            wx.showToast({
              icon: 'none',
              title: '登录成功',
              duration: 1500
            });
          } else {
            // 登录失败
            let title = '';
            if (res.data.result === '密码错误') {
              // 提示密码错误
              title = '密码错误';
            } else if (res.data.result === '用户不存在') {
              // 提示账号不存在
              title = '账号不存在';
            } else {
              // 出现未知错误
              title = '未知错误';
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
    }
  }
})