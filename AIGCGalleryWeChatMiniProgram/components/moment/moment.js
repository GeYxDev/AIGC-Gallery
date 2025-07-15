// 引入事件总线
import EventBus from '../../utils/event-bus'
// 引入媒体预览至浏览界面的路由动画
import { installPointScaleToggleRouteBuilder } from '../../utils/route'

Component({
  properties: {
    // 0: 待加载; 1: 加载中; 2: 无数据; 3: 加载出错
    nextLoadStatus: {
      type: Number,
      value: 0
    }
  },
  data: {
    displayAreaHeight: getApp().globalData.displayAreaHeight,
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
    iuputInfo: {
      showInputBox: false,
      inputBoxPaddingHeight: 0,
      inputCommentContent: '',
      commentItemGroupId: -1
    }
  },
  observers: {
    // 加载状态变化时改变提示
    "nextLoadStatus": function(newNextLoadStatus) {
      let tip = '';
      if (newNextLoadStatus === 1) {
        tip = '加载中';
      } else if (newNextLoadStatus === 2) {
        tip = '无更多数据，请关注更多艺术家或发表更多作品吧';
      } else if (newNextLoadStatus === 3) {
        tip = '加载数据异常，请上滑重试';
      }
      this.setData({ 'nextLoadTip': tip });
    }
  },
  lifetimes: {
    created() {
      // 初始化事件总线
      EventBus.initWorkletEventBus();
      // 初始化媒体预览至浏览界面的路由动画
      installPointScaleToggleRouteBuilder();
    },
    attached() {
      // 加载关注圈数据
      this.openLoadMoment();
    }
  },
  methods: {
    // 首次加载关注圈数据
    openLoadMoment() {
      this.setData({
        'loadInfo.isLoading': true,
        'showFirstLoading': true,
        'showFailLoad': false
      });
      wx.request({
        method: 'POST',
        url: getApp().globalData.baseUrl + 'post/getMoments',
        header: {
          'Content-Type': 'application/x-www-form-urlencoded'
        },
        data: {
          account: getApp().globalData.loginInfo.loginAccount,
          page: this.data.loadInfo.currentPage,
          size: this.data.loadInfo.pageSize
        },
        success: (res) => {
          // 标准化时间格式
          for (const momentItem of res.data.momentList) {
            momentItem.createTime = momentItem.createTime.replace(/\.([^.]*)$/, '');
          }
          this.setData({
            'momentList': res.data.momentList,
            'loadInfo.moreArt': res.data.hasNextPage,
            'showFirstLoading': false
          });
          if (!this.data.loadInfo.moreArt) {
            this.setData({ 'nextLoadStatus': 2 });
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
    tryLoadMoreMoment() {
      this.setData({ 'nextLoadStatus': 1 });
      wx.request({
        method: 'POST',
        url: getApp().globalData.baseUrl + 'post/getMoments',
        header: {
          'Content-Type': 'application/x-www-form-urlencoded'
        },
        data: {
          account: getApp().globalData.loginInfo.loginAccount,
          page: this.data.loadInfo.currentPage,
          size: this.data.loadInfo.pageSize
        },
        success: (res) => {
          // 标准化时间格式
          for (const momentItem of res.data.momentList) {
            momentItem.createTime = momentItem.createTime.replace(/\.([^.]*)$/, '');
          }
          this.setData({
            'momentList': this.data.momentList.concat(res.data.momentList),
            'loadInfo.moreArt': res.data.hasNextPage
          });
          if (this.data.loadInfo.moreArt) {
            this.setData({ 'nextLoadStatus': 0 });
          } else {
            this.setData({ 'nextLoadStatus': 2 });
          }
        },
        fail: () => {
          this.setData({ 'nextLoadStatus': 3 });
        }
      });
    },
    // 页面触底加载更多关注圈数据
    onReachBottom() {
      this.isMoreMomentToLoad();
    },
    // 判断是否继续加载数据
    isMoreMomentToLoad() {
      if (this.data.loadInfo.moreArt && !this.data.loadInfo.isLoading) {
        this.setData({ 'loadInfo.isLoading': true });
        this.setData({ 'loadInfo.currentPage': Number(++this.data.loadInfo.currentPage) });
        this.tryLoadMoreMoment();
        this.setData({ 'loadInfo.isLoading': false });
      }
    },
    // 唤醒输入框
    rouseInputBox(e) {
      // 输入框唤醒震动反馈
      wx.vibrateShort('light');
      this.setData({
        'iuputInfo.showInputBox': true,
        'iuputInfo.commentItemGroupId': e.detail.groupId
      });
    },
    // 处理输入聚焦事件
    planCommentHandle(e) {
      this.setData({ 'iuputInfo.inputBoxPaddingHeight': e.detail.height });
    },
    // 处理输入评论内容
    inputCommentHandle(e) {
      this.setData({ 'iuputInfo.inputCommentContent': e.detail.value });
    },
    // 处理输入取消或结束事件
    cancelCommentHandle() {
      this.setData({
        'iuputInfo.commentItemGroupId': -1,
        'iuputInfo.showInputBox': false,
        'iuputInfo.inputBoxPaddingHeight': 0,
        'iuputInfo.inputCommentContent': ''
      });
    },
    // 发送评论
    sendComment() {
      // 评论发送震动反馈
      wx.vibrateShort('light');
      // 不允许发送空评论
      if (this.data.iuputInfo.inputCommentContent === '') {
        wx.showToast({
          icon: 'none',
          title: '评论内容不能为空哦',
          duration: 2000
        });
        return;
      }
      wx.request({
        method: 'POST',
        url: getApp().globalData.baseUrl + 'comment/addComment',
        header: {
          'Content-Type': 'application/x-www-form-urlencoded'
        },
        data: {
          account: getApp().globalData.loginInfo.loginAccount,
          password: getApp().globalData.loginInfo.loginPassword,
          groupId: this.data.iuputInfo.commentItemGroupId,
          content: this.data.iuputInfo.inputCommentContent
        },
        success: (res) => {
          if (res.data.success) {
            // 更新评论列表
            const newCommentItem = { nickname: getApp().globalData.loginInfo.loginNickname, content: this.data.iuputInfo.inputCommentContent };
            this.selectComponent(`#record-${this.data.iuputInfo.commentItemGroupId}`).updateCommentList(newCommentItem);
            // 收起输入框
            this.cancelCommentHandle();
            // 评论发表成功提示
            wx.showToast({
              icon: 'none',
              title: '评论已发布',
              duration: 1500
            });
          } else {
            let title = '';
            if (res.data.result === '用户验证失败') {
              title = '身份验证失败';
            } else {
              title = '评论失败';
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
    // 删除动态
    deleteMyMoment(e) {
      // 动态删除震动反馈
      wx.vibrateShort('light');
      wx.showModal({
        title: '删除动态',
        content: '确定要删除该动态吗？',
        confirmText: '删除',
        cancelText: '取消',
        success: (res) => {
          if (res.confirm) {
            const momentItemGroupId = e.detail.groupId;
            wx.request({
              method: 'POST',
              url: getApp().globalData.baseUrl + 'post/deleteMoment',
              header: {
                'Content-Type': 'application/x-www-form-urlencoded'
              },
              data: {
                account: getApp().globalData.loginInfo.loginAccount,
                password: getApp().globalData.loginInfo.loginPassword,
                groupId: momentItemGroupId
              },
              success: (res) => {
                if (res.data.success) {
                  // 更新动态列表
                  this.setData({ 'momentList': this.data.momentList.filter(item => item.groupId !== momentItemGroupId) });
                  // 动态删除成功提示
                  wx.showToast({
                    icon: 'none',
                    title: '动态删除成功',
                    duration: 1500
                  });
                } else {
                  let title = '';
                  if (res.data.result === '用户验证失败') {
                    title = '身份验证失败';
                  } else if (res.data.result === '无删除权限') {
                    title = '无删除权限';
                  } else {
                    title = '动态删除失败';
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
      });
    }
  }
})