// 引入MD5加密
import { hexMD5 } from '../../utils/md5'

Page({
  data: {
    topMarginHeight: getApp().globalData.navigationBarHeight,
    loginData: {
      avatar: '/images/default_avatar.png',
      nickname: '',
      account: '',
      password: '',
      confirmPassword: ''
    },
    errors: {
      nickname: '',
      account: '',
      password: '',
      confirmPassword: ''
    },
    passwordVisibility: false,
    confirmPasswordVisibility: false
  },
  // 选择图片作为头像
  selectAvatar() {
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
                    this.setData({ 'loginData.avatar': 'data:image/png;base64,' + res.data });
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
  // 改变密码可见性
  changePasswordVisibility() {
    this.setData({ 'passwordVisibility': !this.data.passwordVisibility });
  },
  // 改变确认密码可见性
  changeConfirmPasswordVisibility() {
    this.setData({ 'confirmPasswordVisibility': !this.data.confirmPasswordVisibility });
  },
  // 昵称输入处理
  nicknameInputChange(e) {
    this.setData({ 'loginData.nickname': e.detail.value });
    this.verifyNickname(e.detail.value);
  },
  // 昵称正确性验证
  verifyNickname(value) {
    let error = '';
    if (!value) {
      error = '昵称不能为空';
    } else if (value.length > 16) {
      error = '昵称长度至多为16位';
    } else if (!/^[A-Za-z0-9\u4e00-\u9fa5]+$/.test(value)) {
      error = '昵称不允许出现该字符';
    }
    this.setData({ 'errors.nickname': error });
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
    } else if (value.length < 8) {
      error = '账号长度至少为8位';
    } else if (value.length > 20) {
      error = '账号长度至多为20位';
    } else if (!/^[a-zA-Z0-9]+$/.test(value)) {
      error = '账号只能由数字和字母组成';
    }
    this.setData({ 'errors.account': error });
  },
  // 密码输入处理
  passwordInputChange(e) {
    this.setData({ 'loginData.password': e.detail.value });
    this.verifyPassword(e.detail.value);
    this.verifyConfirmPassword(this.data.loginData.confirmPassword);
  },
  // 密码正确性验证
  verifyPassword(value) {
    let error = '';
    if (!value) {
      error = '密码不能为空';
    } else if (value.length < 8) {
      error = '密码长度至少为8位';
    } else if (value.length > 16) {
      error = '密码长度至多为16位';
    } else if (!/^[a-zA-Z0-9]+$/.test(value)) {
      error = '密码只能由数字和字母组成';
    } else if (!(/[A-Z]/.test(value) && /[a-z]/.test(value) && /[0-9]/.test(value))) {
      error = '密码必须同时包含大写字母、小写字母和数字';
    }
    this.setData({ 'errors.password': error });
  },
  // 确认密码输入处理
  confirmPasswordInputChange(e) {
    this.setData({ 'loginData.confirmPassword': e.detail.value });
    this.verifyConfirmPassword(e.detail.value);
  },
  // 确认密码准确性验证
  verifyConfirmPassword(value) {
    let error = '';
    if (value !== this.data.loginData.password) {
      error = '密码不一致';
    }
    this.setData({ 'errors.confirmPassword': error });
  },
  // 提交注册操作
  commitRegisterOperation() {
    let errors = this.data.errors;
    if (!this.data.loginData.account) {
        errors.account = '账号不能为空';
    }
    if (!this.data.loginData.password) {
        errors.password = '密码不能为空';
    }
    if (!this.data.loginData.nickname) {
      errors.nickname = '昵称不能为空';
    }
    if (!this.data.loginData.confirmPassword) {
      errors.confirmPassword = '尚未确认密码';
    }
    this.setData({ 'errors': errors });
    // 不存在错误提示时可以提交注册
    if (!errors.account && !errors.nickname && !errors.password && !errors.confirmPassword) {
      // 注册提交操作震动反馈
      wx.vibrateShort('light');
      // md5加密密码
      const encodedPassword = hexMD5(this.data.loginData.password);
      // 向服务器请求注册
      wx.request({
        method: 'POST',
        url: getApp().globalData.baseUrl + 'identity/register',
        header: {
          'Content-Type': 'application/x-www-form-urlencoded'
        },
        data: {
          avatar: this.data.loginData.avatar,
          nickname: this.data.loginData.nickname,
          account: this.data.loginData.account,
          password: encodedPassword
        },
        success: (res) => {
          if (res.data.success === true) {
            // 注册并登录成功
            wx.setStorageSync('loginStatus', 'true');
            wx.setStorageSync('loginAccount', this.data.loginData.account);
            wx.setStorageSync('loginPassword', encodedPassword);
            wx.setStorageSync('loginAvatar', this.data.loginData.avatar);
            wx.setStorageSync('loginNickname', this.data.loginData.nickname);
            getApp().globalData.loginInfo.loginStatus = 'true';
            getApp().globalData.loginInfo.loginAccount = this.data.loginData.account;
            getApp().globalData.loginInfo.loginPassword = encodedPassword;
            getApp().globalData.loginInfo.loginAvatar = this.data.loginData.avatar;
            getApp().globalData.loginInfo.loginNickname = this.data.loginData.nickname;
            // 返回原页面
            wx.navigateBack({ delta: 2 });
            // 注册并登录成功提示
            wx.showToast({
              icon: 'none',
              title: '注册并登录成功',
              duration: 1500
            });
          } else {
            // 注册失败
            let title = '';
            if (res.data.result === '账号已存在') {
              // 提示账号已存在
              title = '账号已存在';
            } else {
              // 出现注册异常
              title = '注册异常';
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