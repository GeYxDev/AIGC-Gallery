Component({
  properties: {
    // 组件索引以及ID
    index: {
      type: Number,
      value: -1,
    },
    // 传入动态数据
    item: {
      type: Object,
      value: {},
    }
  },
  data: {
    account: '',
    convRate: 0.5,
    isSelectLike: false
  },
  lifetimes: {
    attached() {
      // 加载用户账户数据与计算像素转换比例
      this.setData({
        'account': getApp().globalData.loginInfo.loginAccount,
        'convRate': getApp().globalData.windowWidth / 750
      });
    }
  },
  methods: {
    // 媒体浏览选择处理
    onTapMedia(e) {
      if (this.data.item.type === "image") {
        // 若媒体类型为图片
        const { index } = e.currentTarget.dataset;
        // 合成图片列表
        const imageList = Object.values(this.data.item.media).map(item => ({
          src: item.image,
          id: item.workId
        }));
        // 当前选择图片
        const currentImage = imageList[index];
        // 图片列表编码以便于传输
        const encodedImageList = encodeURIComponent(JSON.stringify(imageList));
        // 预览页面跳转
        wx.navigateTo({
          url: `../../pages/preview/preview?rawImageId=${currentImage.id}&rawSourcePageId=${this.getPageId()}&rawMediaList=${encodedImageList}&rawMediaType=${this.data.item.type}`,
          routeType: 'pointScaleToggle'
        });
      } else if (this.data.item.type === "video") {
        // 若媒体类型为视频
        const videoList = [{
          src: this.data.item.media.cover,
          id: this.data.item.media.workId,
          videoLink: this.data.item.media.videoLink,
          mediaRatio: this.data.item.media.mediaRatio
        }];
        // 图片列表编码以便于传输
        const encodedVideoList = encodeURIComponent(JSON.stringify(videoList));
        // 预览页面跳转
        wx.navigateTo({
          url: `../../pages/preview/preview?rawImageId=${this.data.item.media.workId}&rawSourcePageId=${this.getPageId()}&rawMediaList=${encodedVideoList}&rawMediaType=${this.data.item.type}`,
          routeType: 'pointScaleToggle'
        });
      }
    },
    // 动态点赞处理
    handleAddLikeOperation() {
      // 点赞操作震动反馈
      wx.vibrateShort('light');
      if (!this.data.isSelectLike) {
        this.setData({ 'isSelectLike': true });
        wx.request({
          method: 'POST',
          url: getApp().globalData.baseUrl + 'post/addLike',
          header: {
            'Content-Type': 'application/x-www-form-urlencoded'
          },
          data: { groupId: this.data.item.groupId }
        });
      }
    },
    // 动态添加评论处理
    handleAddCommentOperation() {
      this.triggerEvent('addComment', { groupId: this.data.item.groupId });
    },
    // 更新动态评论列表
    updateCommentList(commentItem) {
      let maxCommentId = 0;
      for (const item of this.data.item.comment) {
        if (item.commentId > maxCommentId) {
          maxCommentId = item.commentId;
        }
      }
      const newCommentItem = { ...commentItem, commentId: maxCommentId + 1 };
      this.setData({ 'item.comment': this.data.item.comment.concat(newCommentItem) });
    },
    // 动态删除处理
    handleDeleteMyMomentOperation() {
      this.triggerEvent('deleteMyMoment', { groupId: this.data.item.groupId });
    }
  }
})