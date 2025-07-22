# AI画廊
AI画廊的设计与实现，旨在提供一个具备文本、图像和视频作品生成能力的交互式多模态人工智能生成内容（AIGC）创作平台。AI画廊使用分布式设计，分为前端、后端和计算节点三个部分。

### 前端：微信小程序
AI画廊前端存放在AIGCGalleryWeChatMiniProgram文件夹中，请使用微信开发者工具（[点击下载](https://developers.weixin.qq.com/miniprogram/dev/devtools/download.html)）打开该文件夹，在登录后选择编译即可通过模拟器运行微信小程序。
微信小程序默认后端在本地运行，如有修改需求可通过app.js中的globalData.baseUrl进行修改。
PS：微信小程序上架审核严格，本身也有较多限制，如需修改代码或部署项目请熟读微信小程序开发者文档（[点击打开](https://developers.weixin.qq.com/miniprogram/dev/framework/)）。

### 后端：Spring Boot + Sql Server
AI画廊后端存放在AIGCGalleryBase文件夹中，使用Sql Server存放数据，使用Spring Boot处理数据请求等事务。
Sql Server的建库建表SQL语句如文件createDB.txt所示，如需更换数据库，请同步修改后端中MyBatis和MyBatis-Plus有关数据库连接和操作的代码。
Spring Boot可在Java21上正常运行，若要更换Java版本请先行测试。

### 计算节点：Flask + PyTorch
AI画廊计算节点存放在AIGCGalleryCreator文件夹中，运行creator.py文件即可启动作品创作服务。
运行前请安装需要的包，diffusers包请从根目录下的diffusers文件夹中编译安装，为了显示生成进度以及实现生成中断的操作，该代码与官方代码有所不同。
可运行checkpoints文件夹下的download.py文件，下载所需模型权重。

# 许可证
本项目根据GNU通用公共许可证第三版（GPLv3）获得许可，它包括根据不同条款获得许可的第三方组件，详情请参阅`NOTICE`文件。