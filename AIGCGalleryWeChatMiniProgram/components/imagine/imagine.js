Component({
  data: {
    displayAreaHeight: getApp().globalData.displayAreaHeight,
    scrollAreaHeight: 0,
    // 作品类型
    artType: 'image',
    // 作品列表
    artList: [],
    // 作品生成状态
    isArtGenerating: false,
    // 作品生成提示
    artGeneratePointer: '',
    // 作品生成进度
    artGenerateProgress: 0,
    // 作品保存状态
    isSaveToAlbum: false,
    // 文本生成状态
    textGenerateStatus: {
      cueWord: false,
      theme: false,
      content: false
    },
    // 作品文本数据
    artTextData: {
      cueWord: '',
      theme: '',
      content: ''
    },
    // 图像生成设置
    imageGenerationSettings: {
      modelType: '',
      iterationNumber: ''
    },
    // 视频生成设置
    videoGenerationSettings: {
      modelType: '',
      videoDuration: ''
    },
    // 心跳询问
    heartbeatEnquire: {
      uuid: '',
      timer: null
    },
    // 作品更多菜单
    artMoreMenu: {
      show: false,
      item: [
        { text: '编辑', value: 1 },
        { text: '删除', type: 'warn', value: 2 }
      ],
      triggerId: ''
    },
    // 显示发布艺术
    showReleaseArt: false,
    // 滚动填充高度
    keyboardHeight: 0
  },
  observers: {
    'artTextData.theme, artTextData.content, artList': function() {
      // 必要作品数据存在时显示发布艺术
      if (this.data.artTextData.theme !== '' && this.data.artTextData.content !== '' && this.data.artList.length > 0) {
        this.setData({ 'showReleaseArt': true });
      } else {
        this.setData({ 'showReleaseArt': false });
      }
    }
  },
  lifetimes: {
    attached() {
      // 计算滚动区域显示高度
      this.setData({ 'scrollAreaHeight': getApp().globalData.displayAreaHeight - Math.ceil(getApp().globalData.windowWidth / 7.5) });
    },
    detached() {
      // 清除已设置的定时器防止内存泄漏
      if (this.data.heartbeatEnquire.timer) {
        clearInterval(this.data.heartbeatEnquire.timer);
      }
    }
  },
  methods: {
    // 触发图片预览
    imageArtPreview(e) {
      // 获取需要预览的图片的临时ID
      const tempId = e.currentTarget.dataset.temp;
      const currentImage = this.data.artList.find(item => item.tempId === tempId);
      // 获取当前图像内容
      const currentUrl = currentImage.image;
      // 获取全部图像内容
      const totalUrl = this.data.artList.map(item => item.image);
      // 预览图像
      wx.previewImage({
        urls: totalUrl,
        current: currentUrl,
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
    // 显示图像作品更多菜单
    showImageArtMoreMenu(e) {
      this.setData({
        'artMoreMenu.show': true,
        'artMoreMenu.triggerId': e.currentTarget.dataset.temp
      });
    },
    // 触发视频预览
    videoArtPreview() {
      // base64编码的视频转化为拥有本地路径的文件
      const videoData = this.data.artList[0].video.replace('data:video/mp4;base64,', '');
      const tempFilePath = `${wx.env.USER_DATA_PATH}/temp_video.mp4`;
      wx.getFileSystemManager().writeFile({
        filePath: tempFilePath,
        data: videoData,
        encoding: 'base64',
        success: () => {
          // 预览视频
          wx.previewMedia({
            sources: {
              url: tempFilePath,
              type: 'video',
              poster: this.data.artList[0].cover
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
        },
        fail: () => {
          wx.showToast({
            icon: 'error',
            title: '视频解码失败',
            duration: 2500
          });
        }
      });
    },
    // 显示视频作品更多菜单
    showVideoArtMoreMenu() {
      this.setData({ 'artMoreMenu.show': true });
    },
    // 处理更多作品菜单选择
    handleArtMoreMenuClick(e) {
      let { value } = e.detail;
      if (value === 1) {
        // 编辑作品
        this.editArt();
      } else if (value === 2) {
        // 删除作品
        this.deleteArt();
      } else {
        wx.showToast({
          icon: 'error',
          title: '未知选择',
          duration: 2500
        });
      }
      this.setData({ 'artMoreMenu.show': false });
    },
    // 编辑作品
    editArt() {
      if (this.data.artType === 'image') {
        let imageArt = this.data.artList.find(item => item.tempId === this.data.artMoreMenu.triggerId);
        // base64编码的图像转化为拥有本地路径的文件
        const imageData = image.replace(/^data:image\/\w+;base64,/, '');
        const tempFilePath = `${wx.env.USER_DATA_PATH}/temp_image_${Date.now()}_${index}.png`;
        wx.getFileSystemManager().writeFile({
          filePath: tempFilePath,
          data: imageData,
          encoding: 'base64',
          success: () => {
            // 编辑图像作品
            wx.editImage({
              src: tempFilePath,
              success: (res) => {
                // 获得图像信息
                wx.getImageInfo({
                  src: res.tempFilePath,
                  success: (res) => {
                    const imageWidth = res.width;
                    const imageHeight = res.height;
                    const aspectRatio = imageWidth / imageHeight;
                    imageArt.mediaWidth = imageWidth;
                    imageArt.mediaHeight = imageHeight;
                    imageArt.aspectRatio = aspectRatio;
                    // 将图像转换为base64
                    wx.getFileSystemManager().readFile({
                      filePath: res.path,
                      encoding: 'base64',
                      success: (res) => {
                        imageArt.image = 'data:image/png;base64,' + res.data;
                        // 更新作品列表
                        const updatedArtList = this.data.artList.map(item => {
                          if (item.tempId === this.data.artMoreMenu.triggerId) {
                            return imageArt;
                          }
                          return item;
                        });
                        this.setData({ 'artList': updatedArtList });
                        wx.showToast({
                          icon: 'none',
                          title: '图像编辑成功',
                          duration: 1500
                        });
                      },
                      fail: () => {
                        wx.showToast({
                          icon: 'error',
                          title: '图像转码失败',
                          duration: 2500
                        });
                      }
                    });
                  },
                  fail: () => {
                    wx.showToast({
                      icon: 'error',
                      title: '信息获取失败',
                      duration: 2500
                    });
                  }
                });
              },
              fail: () => {
                wx.showToast({
                  icon: 'error',
                  title: '图像编辑失败',
                  duration: 2500
                });
              }
            });
          },
          fail: () => {
            wx.showToast({
              icon: 'error',
              title: '图像解码失败',
              duration: 2500
            });
          }
        });
      } else if (this.data.artType === 'video') {
        let videoArt = this.data.artList[0];
        // base64编码的视频转化为拥有本地路径的文件
        const videoData = videoArt.video.replace('data:video/mp4;base64,', '');
        const tempFilePath = `${wx.env.USER_DATA_PATH}/temp_video.mp4`;
        wx.getFileSystemManager().writeFile({
          filePath: tempFilePath,
          data: videoData,
          encoding: 'base64',
          success: () => {
            // 编辑视频作品
            wx.openVideoEditor({
              filePath: tempFilePath,
              minDuration: 3,
              maxDuration: 60,
              success: (res) => {
                // 将视频转换为base64
                wx.getFileSystemManager().readFile({
                  filePath: res.tempFilePath,
                  encoding: 'base64',
                  success: (res) => {
                    videoArt.video = 'data:video/mp4;base64,' + res.data;
                    // 更新作品列表
                    this.setData({ 'artList': [videoArt] });
                    wx.showToast({
                      icon: 'none',
                      title: '视频编辑成功',
                      duration: 1500
                    });
                  },
                  fail: () => {
                    wx.showToast({
                      icon: 'error',
                      title: '视频转码失败',
                      duration: 2500
                    });
                  }
                });
              },
              fail: () => {
                wx.showToast({
                  icon: 'error',
                  title: '视频编辑失败',
                  duration: 2500
                });
              }
            });
          },
          fail: () => {
            wx.showToast({
              icon: 'error',
              title: '视频解码失败',
              duration: 2500
            });
          }
        });
      } else {
        wx.showToast({
          icon: 'error',
          title: '未知作品类型',
          duration: 2500
        });
      }
    },
    // 删除作品
    deleteArt() {
      if (this.data.artType === 'image') {
        // 删除图像作品
        wx.showModal({
          title: '删除作品',
          content: '确定要删除该作品吗？',
          confirmText: '删除',
          cancelText: '取消',
          success: (res) => {
            if (res.confirm) {
              // 用户确认删除作品
              const updatedArtList = this.data.artList.filter(item => item.tempId !== this.data.artMoreMenu.triggerId);
              this.setData({ artList: updatedArtList });
              wx.showToast({
                icon: 'none',
                title: '删除成功',
                duration: 1500
              });
            }
          }
        });
      } else if (this.data.artType === 'video') {
        // 删除视频作品
        wx.showModal({
          title: '删除作品',
          content: '确定要删除该作品吗？',
          confirmText: '删除',
          cancelText: '取消',
          success: (res) => {
            if (res.confirm) {
              // 用户确认删除作品
              this.setData({ artList: [] });
              wx.showToast({
                icon: 'none',
                title: '删除成功',
                duration: 1500
              });
            }
          }
        });
      } else {
        wx.showToast({
          icon: 'error',
          title: '未知作品类型',
          duration: 2500
        });
      }
    },
    // 输入时完全显示输入框
    inputAutoScroll(e) {
      const keyboardHeight = e.detail.height;
      const triggerType = e.target.dataset.type;
      // 键盘收起时恢复页面滚动填充
      if (keyboardHeight === 0) {
        this.scrollToInput(0);
        return;
      }
      // 根据触发输入框与遮挡状态决定页面推动距离
      this.createSelectorQuery().select(`.${triggerType}`).boundingClientRect((res) => {
        const windowHeight = getApp().globalData.windowHeight;
        const restHeight = windowHeight - keyboardHeight;
        const inputBottom = res.bottom;
        if (inputBottom > restHeight) {
          const scrollDistance = inputBottom - restHeight;
          this.scrollToInput(keyboardHeight, scrollDistance);
        }
      }).exec();
    },
    // 当前滚动区域滚动位移
    getScrollOffset() {
      return new Promise((resolve) => {
        wx.createSelectorQuery().selectViewport().scrollOffset((res) => resolve(res?.scrollTop || 0)).exec();
      });
    },
    // 触发页面填充与页面滚动
    scrollToInput(keyboardHeight, scrollDistance) {
      this.setData({ 'keyboardHeight': keyboardHeight });
      if (scrollDistance) {
        this.getScrollOffset().then((lastScrollTop) => {
          wx.pageScrollTo({ scrollTop: lastScrollTop + scrollDistance });
        });
      }
    },
    // 确保页面滚动填充恢复
    resetScrollFill() {
      this.scrollToInput(0);
    },
    // 跳转至图像作品生成页面
    selectToCreateImageArt() {
      if (!this.data.isArtGenerating) {
        // 图像生成跳转震动反馈
        wx.vibrateShort('light');
        this.setData({ 'artType': 'image' });
      }
    },
    // 跳转至视频作品生成页面
    selectToCreateVideoArt() {
      if (!this.data.isArtGenerating) {
        // 视频生成跳转震动反馈
        wx.vibrateShort('light');
        this.setData({ 'artType': 'video' });
      }
    },
    // 提示词输入处理
    cueWordInputChange(e) {
      this.setData({ 'artTextData.cueWord': e.detail.value });
    },
    // 主题输入处理
    themeInputChange(e) {
      this.setData({ 'artTextData.theme': e.detail.value });
    },
    // 内容输入处理
    contentInputChange(e) {
      this.setData({ 'artTextData.content': e.detail.value });
    },
    // 作品保存状态切换
    changeSaveToAlbumStatusOperation() {
      this.setData({ 'isSaveToAlbum': !this.data.isSaveToAlbum });
    },
    // 图像生成模型切换处理
    imageGenerateModelChange(e) {
      this.setData({ 'imageGenerationSettings.modelType': e.detail.option });
    },
    // 图像生成迭代次数切换处理
    imageIterationNumberChange(e) {
      this.setData({ 'imageGenerationSettings.iterationNumber': e.detail.option });
    },
    // 视频生成模型切换处理
    videoGenerateModelChange(e) {
      this.setData({ 'videoGenerationSettings.modelType': e.detail.option });
    },
    // 视频生成时长切换处理
    videoGenerateDurationChange(e) {
      this.setData({ 'videoGenerationSettings.videoDuration': e.detail.option });
    },
    // DeepSeek文段生成或润色请求
    requestTextProcess(e) {
      // AI处理触发震动反馈
      wx.vibrateShort('light');
      // 获取触发AI处理的对象
      const type = e.currentTarget.dataset.type;
      let rawText = '';
      let prompt = '';
      let wordLimit = 0;
      if (type === 'cueWord') {
        rawText = this.data.artTextData.cueWord;
        if (this.data.artTextData.theme != '') {
          prompt = this.data.artTextData.theme;
        } else if (this.data.artTextData.content != '') {
          prompt = this.data.artTextData.content;
        }
        wordLimit = 100;
      } else if (type === 'theme') {
        rawText = this.data.artTextData.theme;
        if (this.data.artTextData.content != '') {
          prompt = this.data.artTextData.content;
        } else if (this.data.artTextData.cueWord != '') {
          prompt = this.data.artTextData.cueWord;
        }
        wordLimit = 50;
      } else if (type === 'content') {
        rawText = this.data.artTextData.content;
        if (this.data.artTextData.theme != '') {
          prompt = this.data.artTextData.theme;
        } else if (this.data.artTextData.cueWord != '') {
          prompt = this.data.artTextData.cueWord;
        }
        wordLimit = 500;
      } else {
        wx.showToast({
          icon: 'error',
          title: '未知处理对象',
          duration: 2500
        });
        return;
      }
      // 显示网络请求与AI处理进行动画，并清空对应输入框
      this.setData({
        [`textGenerateStatus.${type}`]: true,
        [`artTextData.${type}`]: ''
      });
      // 发送AI处理请求
      wx.request({
        method: 'POST',
        url: getApp().globalData.baseUrl + 'api/t2t',
        header: {
          'Content-Type': 'application/x-www-form-urlencoded'
        },
        data: {
          account: getApp().globalData.loginInfo.loginAccount,
          password: getApp().globalData.loginInfo.loginPassword,
          text: rawText,
          wordLimit: wordLimit,
          textType: type,
          prompt: prompt
        },
        success: (res) => {
          if (res.data.success === true) {
            // AI处理结果返回成功
            this.setData({ [`artTextData.${type}`]: res.data.result });
            wx.showToast({
              icon: 'none',
              title: '您的文案完成啦～',
              duration: 1500
            });
          } else {
            // AI处理结果返回失败
            this.setData({ [`artTextData.${type}`]: rawText });
            let title = '';
            if (res.data.result === '密码错误') {
              // 提示身份验证失败
              title = '身份验证失败';
            } else if (res.data.result === '用户不存在') {
              // 提示用户不存在
              title = '用户不存在';
            } else if (res.data.result === '服务启动失败') {
              // 提示服务未开启
              title = '服务未开启';
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
          // 返还用户输入的文字
          this.setData({ [`artTextData.${type}`]: rawText });
          wx.showToast({
            icon: 'error',
            title: '网络异常',
            duration: 2500
          });
        },
        complete: () => {
          // 解除网络请求与AI处理进行动画
          this.setData({ [`textGenerateStatus.${type}`]: false });
        }
      });
    },
    // 发送作品生成参数与生成请求
    sendArtCreateParameter() {
      // 已有作品在生成状态时不允许提交生成请求
      if (this.data.isArtGenerating) {
        wx.showToast({
          icon: 'none',
          title: '作品正在生成中',
          duration: 1500
        });
        return;
      }
      // 作品生成处理触发震动反馈
      wx.vibrateShort('light');
      // 未填提示词时驳回请求
      if (this.data.artTextData.cueWord === '') {
        wx.showToast({
          icon: 'none',
          title: '请填写提示词',
          duration: 1500
        });
        return;
      }
      // 生成请求链接与参数表
      let requestUrl = '';
      let requestParameter = {};
      if (this.data.artType === 'image') {
        requestUrl = getApp().globalData.baseUrl + 'api/t2i';
        requestParameter = {
          word: this.data.artTextData.cueWord,
          model: this.data.imageGenerationSettings.modelType,
          iteration: this.data.imageGenerationSettings.iterationNumber
        };
      } else if (this.data.artType === 'video') {
        requestUrl = getApp().globalData.baseUrl + 'api/t2v';
        requestParameter = {
          word: this.data.artTextData.cueWord,
          model: this.data.videoGenerationSettings.modelType,
          duration: this.data.videoGenerationSettings.videoDuration
        };
      } else {
        wx.showToast({
          icon: 'error',
          title: '未知生成类型',
          duration: 2500
        });
        return;
      }
      // 设置作品正在生成状态
      this.setData({ 
        'isArtGenerating': true,
        'artGeneratePointer': '处理中……'
       });
      // 发送作品生成请求
      wx.request({
        method: 'POST',
        url: requestUrl,
        header: {
          'Content-Type': 'application/x-www-form-urlencoded'
        },
        data: {
          ...requestParameter,
          account: getApp().globalData.loginInfo.loginAccount,
          password: getApp().globalData.loginInfo.loginPassword
        },
        success: (res) => {
          if (res.data.success === true) {
            // 作品生成请求进入处理序列中
            this.setData({ 'heartbeatEnquire.uuid': res.data.result });
            // 根据作品生成类型延迟激活作品生成进度询问操作
            let delayTime = 3000;
            if (this.data.artType === 'image') {
              delayTime = 6000;
            } else if (this.data.artType === 'video') {
              delayTime = 16000;
            }
            setTimeout(() => {
              // 激活作品生成进度询问操作
              this.heartbeatEnquireProgress();
            }, delayTime);
          } else {
            // 解除作品正在生成状态
            this.setData({
              'isArtGenerating': false,
              'artGeneratePointer': ''
            });
            let title = '';
            if (res.data.result === '密码错误') {
              // 提示身份验证失败
              title = '身份验证失败';
            } else if (res.data.result === '用户不存在') {
              // 提示用户不存在
              title = '用户不存在';
            } else if (res.data.result === '服务启动失败') {
              // 提示服务未开启
              title = '服务未开启';
            } else if (res.data.result === '作品生成失败') {
              // 提示作品生成失败
              title = '作品生成失败';
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
          // 解除作品正在生成状态
          this.setData({
            'isArtGenerating': false,
            'artGeneratePointer': ''
          });
          wx.showToast({
            icon: 'error',
            title: '网络异常',
            duration: 2500
          });
        }
      });
    },
    // 作品生成进度持续询问
    heartbeatEnquireProgress() {
      // 清除已设置的定时器
      if (this.data.heartbeatEnquire.timer) {
        clearInterval(this.data.heartbeatEnquire.timer);
      }
      // 设定新的计时器以实现每秒一次的作品生成进度询问
      const timer = setInterval(() => {
        wx.request({
          method: 'POST',
          timeout: 16000,
          url: getApp().globalData.baseUrl + 'api/enquire',
          header: {
            'Content-Type': 'application/x-www-form-urlencoded'
          },
          data: { uuid: this.data.heartbeatEnquire.uuid },
          success: (res) => {
            // 获得作品生成反馈后不响应后续请求结果
            if (this.data.isArtGenerating) {
              if (res.data.success === true) {
                // 请求响应成功且作品生成状态正常
                if (res.data.result.status === 'waiting') {
                  // 作品生成等待中
                  this.setData({
                    'artGeneratePointer': '等待中……',
                    'artGenerateProgress': 0
                  });
                } else if (res.data.result.status === 'creating') {
                  // 作品生成中
                  this.setData({
                    'artGeneratePointer': `生成中：${parseInt(res.data.result.progress)}%`,
                    'artGenerateProgress': parseInt(res.data.result.progress)
                  });
                } else if (res.data.result.status === 'complete') {
                  // 作品生成完成
                  clearInterval(timer);
                  this.setData({
                    'isArtGenerating': false,
                    'artGeneratePointer': '',
                    'artGenerateProgress': 0,
                    'artList': this.data.artList.concat(res.data.result.work)
                  });
                  wx.showToast({
                    icon: 'none',
                    title: '作品完成啦',
                    duration: 2500
                  });
                } else {
                  // 作品生成状态反馈异常
                  clearInterval(timer);
                  this.setData({
                    'isArtGenerating': false,
                    'artGeneratePointer': '',
                    'artGenerateProgress': 0
                  });
                  wx.showToast({
                    icon: 'error',
                    title: '作品生成失败',
                    duration: 2500
                  });
                }
              } else {
                // 请求响应成功但作品生成状态异常
                clearInterval(timer);
                this.setData({
                  'isArtGenerating': false,
                  'artGeneratePointer': '',
                  'artGenerateProgress': 0
                });
                wx.showToast({
                  icon: 'error',
                  title: '作品生成失败',
                  duration: 2500
                });
              }
            }
          },
          fail: () => {
            // 获得作品生成反馈后不响应后续请求结果
            if (this.data.isArtGenerating) {
              clearInterval(timer);
              this.setData({
                'isArtGenerating': false,
                'artGeneratePointer': '',
                'artGenerateProgress': 0
              });
              wx.showToast({
                icon: 'error',
                title: '网络异常',
                duration: 2500
              });
            }
          }
        });
      }, 2000);
      this.setData({ 'heartbeatEnquire.timer': timer });
    },
    // 发布作品
    handleReleaseOperation() {
      // 存在进行中的生成任务时不允许发布作品
      if (isArtGenerating || textGenerateStatus.cueWord || textGenerateStatus.theme || textGenerateStatus.content) {
        wx.showToast({
          icon: 'none',
          title: '请等待生成完成哦～',
          duration: 2500
        });
        return;
      }
      // 作品发布触发震动反馈
      wx.vibrateShort('light');
      // 作品发布中提示
      wx.showToast({
        icon: 'loading',
        title: '作品发布中',
        duration: 100000,
        mask: true
      });
      // 发布作品
      wx.request({
        method: 'POST',
        url: getApp().globalData.baseUrl + 'post/releaseMoment',
        header: {
          'Content-Type': 'application/x-www-form-urlencoded'
        },
        data: {
          account: getApp().globalData.loginInfo.loginAccount,
          password: getApp().globalData.loginInfo.loginPassword,
          theme: this.data.artTextData.theme,
          content: this.data.artTextData.content,
          artList: JSON.stringify(this.data.artList),
          artType: this.data.artType
        },
        success: (res) => {
          if (res.data.success === true) {
            // 作品发布成功
            if (this.data.isSaveToAlbum) {
              // 将生成的作品保存至相册
              if (this.data.artType === 'image') {
                const totalImage = this.data.artList.map(item => item.image);
                totalImage.forEach((image, index) => {
                  const imageData = image.replace(/^data:image\/\w+;base64,/, '');
                  const tempFilePath = `${wx.env.USER_DATA_PATH}/temp_image_${Date.now()}_${index}.png`;
                  wx.getFileSystemManager().writeFile({
                    filePath: tempFilePath,
                    data: imageData,
                    encoding: 'base64',
                    success: () => {
                      wx.saveImageToPhotosAlbum({ filePath: tempFilePath });
                    }
                  });
                });
              } else if (this.data.artType === 'video') {
                const videoData = this.data.artList[0].video.replace('data:video/mp4;base64,', '');
                const tempFilePath = `${wx.env.USER_DATA_PATH}/temp_video.mp4`;
                wx.getFileSystemManager().writeFile({
                  filePath: tempFilePath,
                  data: videoData,
                  encoding: 'base64',
                  success: () => {
                    wx.saveVideoToPhotosAlbum({ filePath: tempFilePath });
                  }
                });
              }
            }
            // 清除已有作品内容
            this.setData({
              'artList': [],
              'artTextData.cueWord': false,
              'artTextData.theme': false,
              'artTextData.content': false
            });
            wx.hideToast();
            wx.showToast({
              icon: 'none',
              title: '作品发布成功',
              duration: 1500
            });
          } else {
            // 作品发布失败
            wx.hideToast();
            let title = '';
            if (res.data.result === '密码错误') {
              // 提示身份验证失败
              title = '身份验证失败';
            } else if (res.data.result === '用户不存在') {
              // 提示用户不存在
              title = '用户不存在';
            } else if (res.data.result === '作品发布失败') {
              // 提示作品发布失败
              title = '作品发布失败';
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
          wx.hideToast();
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