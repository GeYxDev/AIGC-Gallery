// 引入手势与动画相关方法
import { Curves, CurveAnimation, lerp, clamp, GestureState } from '../../utils/route'

Component({
  properties: {
    index: {
      type: Number,
      value: -1
    },
    groupId: {
      type: Number,
      value: -1
    },
    account: {
      type: String,
      value: ''
    },
    nickname: {
      type: String,
      value: ''
    },
    avatar: {
      type: String,
      value: ''
    },
    type: {
      type: String,
      value: 'image'
    },
    media: {
      type: String,
      value: ''
    },
    theme: {
      type: String,
      value: ''
    },
    aspectRatio: {
      type: Number,
      value: 1.0
    },
    likes: {
      type: Number,
      value: 0
    }
  },
  data: {
    navigationBarHeight: getApp().globalData.navigationBarHeight,
    menuButtonTopPaddingHeight: wx.getMenuButtonBoundingClientRect().top,
    menuButtonHeight: wx.getMenuButtonBoundingClientRect().height,
    pageWidth: getApp().globalData.windowWidth,
    safeAreaBottomHeight: getApp().globalData.safeAreaBottomHeight,
    menuButtonLeftBorder: 0,
    scrollAreaHeight: 0,
    isSelectFollow: false,
    isSelectLike: false,
    post: {},
    comment: {},
    commentUsualInfo: '',
    iuputInfo: {
      showInputBox: false,
      inputBoxPaddingHeight: 0,
      inputCommentContent: ''
    }
  },
  lifetimes: {
    created() {
      // 动画标点
      this.startX = wx.worklet.shared(0)
      this.startY = wx.worklet.shared(0)
      this.transX = wx.worklet.shared(0)
      this.transY = wx.worklet.shared(0)
      this.isInteracting = wx.worklet.shared(false)
    },
    attached() {
      // 计算菜单按钮宽度和滚动区域高度
      this.setData({
        'menuButtonLeftBorder': getApp().globalData.windowWidth - wx.getMenuButtonBoundingClientRect().left,
        'scrollAreaHeight': getApp().globalData.windowHeight - this.data.navigationBarHeight - this.data.safeAreaBottomHeight - 40
      });
      // 加载作品详细信息
      this.loadArtDetail()
      // 加载评论信息
      this.loadCommentInfo()
      // 加载关注信息
      this.loadFollowInfo()
      // 路由信息
      this.customRouteContext = wx.router?.getRouteContext(this)
      const { 
        primaryAnimation,
        primaryAnimationStatus,
        userGestureInProgress,
        shareEleTop
      } = this.customRouteContext || {}
      // 根据进入或返回情况使用不同曲线
      const _curvePrimaryAnimation = CurveAnimation({
        animation: primaryAnimation,
        animationStatus: primaryAnimationStatus,
        curve: wx.worklet.Easing.in(Curves.fastOutSlowIn),
        reverseCurve: wx.worklet.Easing.out(Curves.fastOutSlowIn)
      });
      // 主题与正文部分动画
      this.applyAnimatedStyle('.detail-content', () => {
        'worklet'
        return {
          opacity: _curvePrimaryAnimation.value
        }
      });
      // 导航栏部分动画
      this.applyAnimatedStyle('.navigation-bar', () => {
        'worklet'
        return {
          opacity: _curvePrimaryAnimation.value
        }
      });
      // 互动部分动画
      this.applyAnimatedStyle('.detail-func', () => {
        'worklet'
        return {
          opacity: _curvePrimaryAnimation.value
        }
      });
      // 评论部分动画
      this.applyAnimatedStyle('.detail-comment', () => {
        'worklet'
        return {
          opacity: _curvePrimaryAnimation.value
        }
      });
      // 细节页面整体进入与返回动画
      this.applyAnimatedStyle('.detail-container', () => {
        'worklet'
        if (userGestureInProgress.value && globalThis['RouteCardSrcRect'] && globalThis['RouteCardSrcRect'].value != undefined) {
          const begin = globalThis['RouteCardSrcRect'].value
          const end = globalThis['RouteCardDestRect'].value
          const t = 1 - _curvePrimaryAnimation.value
          const shareEleX = lerp(begin.left, end.left, t)
          const shareEleY = lerp(begin.top, end.top, t)
          const shareEleW = lerp(begin.width, end.width, t)
          const scale = shareEleW / this.data.pageWidth
          const transX = shareEleX
          // 比例换算使图片顶边对齐
          const transY = shareEleY - shareEleTop.value * scale
          // 透明度变换
          const alpha = 1 - Curves.easeInExpo(t)
          return {
            transform: `translateX(${ transX }px) translateY(${ transY }px) scale(${ scale })`,
            transformOrigin: '0 0',
            backgroundColor: `rgba(255, 255, 255, ${ alpha })`
          }
        }
        // 页面跟随手势移动
        const transX = this.transX.value
        const transY = this.transY.value
        // 依据横坐标位移比例缩放
        const scale = clamp(1 - transX / this.data.pageWidth * 0.5, 0, 1)
        // 透明度变换
        const normalizedTransX = Math.abs(transX) / this.data.pageWidth
        const alpha = 1 - Curves.easeInExpo(normalizedTransX)
        return {
          transform: `translateX(${ transX }px) translateY(${ transY }px) scale(${ scale })`,
          transformOrigin: '50% 50%',
          backgroundColor: `rgba(255, 255, 255, ${ alpha })`
        }
      }, {
        immediate: false
      });
    },
    detached() {
      // 发送作品点赞数更新
      this.sendArtLikeUpdate()
    }
  },
  methods: {
    // 加载作品详细信息
    loadArtDetail() {
      wx.request({
        method: 'POST',
        url: getApp().globalData.baseUrl + 'detail/getArtDetail',
        header: {
          'Content-Type': 'application/x-www-form-urlencoded'
        },
        data: { groupId: this.data.groupId },
        success: (res) => {
          if (this.data.type === 'image') {
            // 去除workId最小的媒体
            let minWorkId = Infinity;
            for (const item of res.data.mediaDetailList) {
              if (item.workId < minWorkId) {
                  minWorkId = item.workId;
              }
            }
            res.data.mediaDetailList = res.data.mediaDetailList.filter(item => item.workId !== minWorkId);
          }
          this.setData({ 'post': res.data });
        },
        fail: () => {
          wx.showToast({
            icon: 'error',
            title: '加载失败',
            duration: 2500
          });
        }
      });
    },
    // 加载评论信息
    loadCommentInfo() {
      wx.request({
        method: 'POST',
        url: getApp().globalData.baseUrl + 'detail/getComments',
        header: {
          'Content-Type': 'application/x-www-form-urlencoded'
        },
        data: { groupId: this.data.groupId },
        success: (res) => {
          if (Object.keys(res.data).length > 0) {
            // 标准化时间格式
            for (const commentItem of res.data) {
              commentItem.createTime = commentItem.createTime.replace(/\.([^.]*)$/, '');
            }
            this.setData({ 'comment': res.data });
          } else {
            this.setData({ 'commentUsualInfo': '暂时没有发现评论存在的迹象' });
          }
        },
        fail: () => {
          this.setData({ 'commentUsualInfo': '评论加载出错，请退出重试' });
        }
      });
    },
    // 加载关注信息
    loadFollowInfo() {
      wx.request({
        method: 'POST',
        url: getApp().globalData.baseUrl + 'follow/isFollowed',
        header: {
          'Content-Type': 'application/x-www-form-urlencoded'
        },
        data: {
          account: getApp().globalData.loginInfo.loginAccount,
          followedAccount: this.data.account
        },
        success: (res) => {
          if (res.data.success) {
            this.setData({ 'isSelectFollow': res.data.result });
          }
        }
      });
    },
    // 处理关注操作
    handleFollowOperation() {
      // 关注操作震动反馈
      wx.vibrateShort('light');
      // 若未登录需要先进行登录操作
      if (getApp().globalData.loginInfo.loginStatus === "false") {
        wx.navigateTo({ url: "/pages/login/login" });
        return;
      }
      let urlContent = '';
      if (this.data.isSelectFollow) {
        // 若已关注则选择取消关注操作
        urlContent = getApp().globalData.baseUrl + 'follow/cancelFollow';
      } else {
        // 若未关注则进行关注操作
        urlContent = getApp().globalData.baseUrl + 'follow/addFollow';
      }
      wx.request({
        method: 'POST',
        url: urlContent,
        header: {
          'Content-Type': 'application/x-www-form-urlencoded'
        },
        data: {
          account: getApp().globalData.loginInfo.loginAccount,
          password: getApp().globalData.loginInfo.loginPassword,
          followedAccount: this.data.account
        },
        success: (res) => {
          if (res.data.success) {
            this.setData({ 'isSelectFollow': !this.data.isSelectFollow });
          } else {
            let title = '';
            if (res.data.result === '用户验证失败') {
              title = '身份验证失败';
            } else {
              title = '操作失败';
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
    // 处理作品点赞事件
    handleAddLikeOperation() {
      // 点赞操作震动反馈
      wx.vibrateShort('light');
      // 若未登录需要先进行登录操作
      if (getApp().globalData.loginInfo.loginStatus === "false") {
        wx.navigateTo({ url: "/pages/login/login" });
        return;
      }
      this.setData({
        'likes': this.data.isSelectLike ? this.data.likes - 1 : this.data.likes + 1,
        'isSelectLike': !this.data.isSelectLike
      });
    },
    // 发送作品点赞数更新
    sendArtLikeUpdate() {
      if (this.data.isSelectLike) {
        wx.request({
          method: 'POST',
          url: getApp().globalData.baseUrl + 'post/addLike',
          header: {
            'Content-Type': 'application/x-www-form-urlencoded'
          },
          data: { groupId: this.data.groupId }
        });
      }
    },
    // 唤醒输入框
    rouseInputBox() {
      // 输入框唤醒震动反馈
      wx.vibrateShort('light');
      this.setData({ 'iuputInfo.showInputBox': true });
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
        'iuputInfo.showInputBox': false,
        'iuputInfo.inputBoxPaddingHeight': 0
      });
    },
    // 更新动态评论列表
    updateCommentList() {
      // 获得评论临时序列号
      let maxCommentId = 0;
      for (const item of this.data.comment) {
        if (item.commentId > maxCommentId) {
          maxCommentId = item.commentId;
        }
      }
      const commentId = maxCommentId + 1;
      // 获得评论发布临时时间
      const now = new Date();
      const year = String(now.getFullYear()).padStart(2, '0');
      const month = String(now.getMonth() + 1).padStart(2, '0');
      const day = String(now.getDate()).padStart(2, '0');
      const hours = String(now.getHours()).padStart(2, '0');
      const minutes = String(now.getMinutes()).padStart(2, '0');
      const seconds = String(now.getSeconds()).padStart(2, '0');
      const date = `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
      // 合成新评论项
      const commentItem = {
        nickname: getApp().globalData.loginInfo.loginNickname,
        avatar: getApp().globalData.loginInfo.loginAvatar,
        commentId: commentId,
        content: this.data.iuputInfo.inputCommentContent,
        createTime: date
      };
      this.setData({ 'comment': this.data.comment.concat(commentItem) });
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
          groupId: this.data.groupId,
          content: this.data.iuputInfo.inputCommentContent
        },
        success: (res) => {
          if (res.data.success) {
            // 更新评论列表
            this.updateCommentList();
            // 清空评论输入框
            this.setData({ 'iuputInfo.inputCommentContent': '' });
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
    // 处理返回手势
    handleReturnGesture(e) {
      'worklet'
      const {
        startUserGesture,
        stopUserGesture,
        primaryAnimation,
        didPop
      } = this.customRouteContext
      // 手势开始
      if (e.state === GestureState.BEGIN) {
        this.startX.value = e.absoluteX
        this.startY.value = e.absoluteY
      // 手势进行过程
      } else if (e.state === GestureState.ACTIVE) {
        // 手势向右滑动
        if (e.deltaX > 0 && !this.isInteracting.value) {
          this.isInteracting.value = true
        }
        if (!this.isInteracting.value) return
        const transX = e.absoluteX - this.startX.value
        // 最小转移边界
        const transLowerBound = -1/3 * this.data.pageWidth
        // 最大转移边界
        const transUpperBound = 2/3 * this.data.pageWidth
        this.transX.value = clamp(transX, transLowerBound, transUpperBound)
        this.transY.value = e.absoluteY - this.startY.value
      // 手势结束或取消
      } else if (e.state === GestureState.END || e.state === GestureState.CANCELLED) {
        if (!this.isInteracting.value) return
        this.isInteracting.value = false
        let shouldFinish = false
        if (e.velocityX > 500 || this.transX.value / this.data.pageWidth > 0.25) {
          shouldFinish = true
        }
        if (shouldFinish) {
          startUserGesture()
          primaryAnimation.value = wx.worklet.timing(0.0, {
            duration: 180,
            easing: wx.worklet.Easing.linear
          }, () => {
            'worklet'
            stopUserGesture()
            didPop()
          });
        } else {
          this.transX.value = wx.worklet.timing(0.0, { duration: 100 })
          this.transY.value = wx.worklet.timing(0.0, { duration: 100 })
        }
      }
    }
  }
})