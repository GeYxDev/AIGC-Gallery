// 引入MD5加密
import { hexMD5 } from '../../utils/md5'

Component({
  properties: {
    modifyType: {
      type: String,
      value: 'password'
    }
  },
  data: {
    topMarginHeight: getApp().globalData.navigationBarHeight,
    loginData: {
      oldPassword: '',
      password: '',
      confirmPassword: '',
      nickname: ''
    },
    errors: {
      oldPassword: '',
      password: '',
      confirmPassword: '',
      nickname: ''
    },
    oldPasswordVisibility: false,
    passwordVisibility: false,
    confirmPasswordVisibility: false
  },
  methods: {
    // 改变旧密码可见性
    changeOldPasswordVisibility() {
      this.setData({ 'oldPasswordVisibility': !this.data.oldPasswordVisibility });
    },
    // 改变密码可见性
    changePasswordVisibility() {
      this.setData({ 'passwordVisibility': !this.data.passwordVisibility });
    },
    // 改变确认密码可见性
    changeConfirmPasswordVisibility() {
      this.setData({ 'confirmPasswordVisibility': !this.data.confirmPasswordVisibility });
    },
    // 旧密码输入处理
    oldPasswordInputChange(e) {
      this.setData({ 'loginData.oldPassword': e.detail.value });
      this.verifyOldPassword(e.detail.value);
    },
    // 旧密码正确性验证
    verifyOldPassword(value) {
      let error = '';
      if (!value) {
        error = '旧密码不能为空';
      }
      this.setData({ 'errors.oldPassword': error });
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
    // 提交修改操作
    commitModifyOperation() {
      let errors = this.data.errors;
      if (this.data.modifyType === 'password') {
        // 若为修改密码
        if (!this.data.loginData.oldPassword) {
            errors.account = '旧密码不能为空';
        }
        if (!this.data.loginData.password) {
            errors.password = '密码不能为空';
        }
        if (!this.data.loginData.confirmPassword) {
          errors.confirmPassword = '尚未确认密码';
        }
      } else {
        // 若为修改昵称
        if (!this.data.loginData.nickname) {
          errors.nickname = '昵称不能为空';
        }
      }
      this.setData({ 'errors': errors });
      // 不存在错误提示时可以提交修改
      if (this.data.modifyType === 'password') {
        // 若为修改密码
        if (!errors.oldPassword && !errors.password && !errors.confirmPassword) {
          // 密码修改震动反馈
          wx.vibrateShort('light');
          // md5加密密码
          const encodedOldPassword = hexMD5(this.data.loginData.oldPassword);
          const encodedPassword = hexMD5(this.data.loginData.password);
          // 向服务器请求修改
          wx.request({
            method: 'POST',
            url: getApp().globalData.baseUrl + 'identity/modifyPassword',
            header: {
              'Content-Type': 'application/x-www-form-urlencoded'
            },
            data: {
              account: getApp().globalData.loginInfo.loginAccount,
              oldPassword: encodedOldPassword,
              newPassword: encodedPassword
            },
            success: (res) => {
              if (res.data.success === true) {
                // 密码修改成功
                wx.setStorageSync('loginPassword', encodedPassword);
                getApp().globalData.loginInfo.loginPassword = encodedPassword;
                // 返回原页面
                wx.navigateBack();
                // 密码修改成功提示
                wx.showToast({
                  icon: 'none',
                  title: '密码修改成功',
                  duration: 1500
                });
              } else {
                // 密码修改失败
                let title = '';
                if (res.data.result === '账号不存在') {
                  // 提示账号不存在
                  title = '账号不存在';
                } else if (res.data.result === '旧密码错误') {
                  // 提示旧密码错误
                  title = '旧密码错误';
                } else {
                  // 出现密码修改异常
                  title = '密码修改异常';
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
      } else {
        // 若为修改昵称
        if (!errors.nickname) {
          // 昵称修改震动反馈
          wx.vibrateShort('light');
          // 向服务器请求修改
          wx.request({
            method: 'POST',
            url: getApp().globalData.baseUrl + 'identity/modifyNickname',
            header: {
              'Content-Type': 'application/x-www-form-urlencoded'
            },
            data: {
              account: getApp().globalData.loginInfo.loginAccount,
              password: getApp().globalData.loginInfo.loginPassword,
              nickname: this.data.loginData.nickname
            },
            success: (res) => {
              if (res.data.success === true) {
                // 昵称修改成功
                wx.setStorageSync('loginNickname', this.data.loginData.nickname);
                getApp().globalData.loginInfo.loginNickname = this.data.loginData.nickname;
                // 返回原页面
                wx.navigateBack();
                // 昵称修改成功提示
                wx.showToast({
                  icon: 'none',
                  title: '昵称修改成功',
                  duration: 1500
                });
              } else {
                // 昵称修改失败
                let title = '';
                if (res.data.result === '账号不存在') {
                  // 提示账号不存在
                  title = '账号不存在';
                } else if (res.data.result === '密码错误') {
                  // 提示身份验证失败
                  title = '身份验证失败';
                } else {
                  // 出现昵称修改异常
                  title = '昵称修改异常';
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
    }
  }
})