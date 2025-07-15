// 引入路由动画
import { installScaleTransitionRouteBuilder } from '../../utils/route'

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
    cardWidth: 0,
    artList: [],
    loadInfo: {
      // 当前页
      currentPage: 1,
      // 单页条数
      pageSize: 8,
      // 是否还有更多数据
      moreArt: true,
      // 是否在加载
      isLoading: false
    },
    showFirstLoading: false,
    showFailLoad: false,
    nextLoadTip: ''
  },
  observers: {
    // 加载状态变化时改变提示
    "nextLoadStatus": function(newNextLoadStatus) {
      let tip = '';
      if (newNextLoadStatus === 1) {
        tip = '加载中';
      } else if (newNextLoadStatus === 2) {
        tip = '无更多数据';
      } else if (newNextLoadStatus === 3) {
        tip = '加载数据异常，请上滑重试';
      }
      this.setData({ 'nextLoadTip': tip });
    }
  },
  lifetimes: {
    created() {
      // 计算卡片宽度
      this.setData({ 'cardWidth': (getApp().globalData.windowWidth - 18) / 2 });
      // 创建时即加载画廊展示数据
      this.openLoadArt();
      // 引入自定义路由
      installScaleTransitionRouteBuilder();
    }
  },
  methods: {
    // 首次加载画廊展示数据
    openLoadArt() {
      this.setData({
        'loadInfo.isLoading': true,
        'showFirstLoading': true,
        'showFailLoad': false
      });
      wx.request({
        method: 'POST',
        url: getApp().globalData.baseUrl + 'roam/getCards',
        header: {
          'Content-Type': 'application/x-www-form-urlencoded'
        },
        data: {
          page: this.data.loadInfo.currentPage,
          size: this.data.loadInfo.pageSize
        },
        success: (res) => {
          this.setData({
            'artList': res.data.cardList,
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
    // 加载画廊展示数据
    tryLoadMoreArt() {
      this.setData({ 'nextLoadStatus': 1 });
      wx.request({
        method: 'POST',
        url: getApp().globalData.baseUrl + 'roam/getCards',
        header: {
          'Content-Type': 'application/x-www-form-urlencoded'
        },
        data: {
          page: this.data.loadInfo.currentPage,
          size: this.data.loadInfo.pageSize
        },
        success: (res) => {
          this.setData({
            'artList': this.data.artList.concat(res.data.cardList),
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
    // 页面触底加载更多画廊展示数据
    onReachBottom() {
      this.isMoreArtToLoad();
    },
    // 判断是否继续加载数据
    isMoreArtToLoad() {
      if (this.data.loadInfo.moreArt && !this.data.loadInfo.isLoading) {
        this.setData({ 'loadInfo.isLoading': true });
        this.setData({ 'loadInfo.currentPage': Number(++this.data.loadInfo.currentPage) });
        this.tryLoadMoreArt();
        this.setData({ 'loadInfo.isLoading': false });
      }
    }
  }
});