//package util;
//
//import com.baomidou.mybatisplus.annotation.DbType;
//import com.baomidou.mybatisplus.generator.AutoGenerator;
//import com.baomidou.mybatisplus.generator.config.*;
//import com.baomidou.mybatisplus.generator.config.rules.NamingStrategy;
//
//// 请移至java文件夹下使用
//public class CodeGenerator {
//    public static void main(String[] args) {
//        // 创建代码生成器
//        AutoGenerator mpg = new AutoGenerator();
//
//        // 全局配置
//        GlobalConfig gc = new GlobalConfig();
//        gc.setOutputDir("src/main/java"); // 输出目录
//        gc.setAuthor("NeYx"); // 作者
//        gc.setOpen(false); // 是否打开输出目录
//        gc.setFileOverride(true); // 是否覆盖已有文件
//        gc.setServiceName("%sService"); // 设置 Service 文件名
//        gc.setServiceImplName("%sServiceImpl"); // 设置 ServiceImpl 文件名
//        mpg.setGlobalConfig(gc);
//
//        // 数据源配置
//        DataSourceConfig dsc = new DataSourceConfig();
//        dsc.setDbType(DbType.SQL_SERVER); // 数据库类型
//        dsc.setDriverName("com.microsoft.sqlserver.jdbc.SQLServerDriver");
//        dsc.setUrl("jdbc:sqlserver://localhost:1433;databaseName=AIGCGalleryDB;encrypt=false");
//        dsc.setUsername("sa");
//        dsc.setPassword("12345678");
//        mpg.setDataSource(dsc);
//
//        // 包配置
//        PackageConfig pc = new PackageConfig();
//        pc.setParent("dao");
//        pc.setEntity("entity"); // 实体类包名
//        pc.setMapper("mapper"); // Mapper 接口包名
//        pc.setService("service"); // Service 接口包名
//        pc.setServiceImpl("service.impl"); // Service 实现包名
//        pc.setController("controller"); // Controller 包名
//        mpg.setPackageInfo(pc);
//
//        // 策略配置
//        StrategyConfig strategy = new StrategyConfig();
//        strategy.setNaming(NamingStrategy.underline_to_camel); // 数据库表映射到实体的命名策略
//        strategy.setColumnNaming(NamingStrategy.underline_to_camel); // 数据库表字段映射到实体的命名策略
//        strategy.setInclude("Users", "Follows", "Groups", "Works", "Videos", "Images", "Comments"); // 需要生成的表名
//        strategy.setControllerMappingHyphenStyle(true); // Controller 映射路径使用连字符风格
//        mpg.setStrategy(strategy);
//
//        // 执行生成
//        mpg.execute();
//    }
//}