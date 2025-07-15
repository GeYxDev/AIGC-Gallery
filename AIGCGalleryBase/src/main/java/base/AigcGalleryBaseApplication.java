package base;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.transaction.annotation.EnableTransactionManagement;

@SpringBootApplication
@EnableTransactionManagement
@MapperScan("dao.mapper")
@ComponentScan("dao.*")
@ComponentScan("func.*")
public class AigcGalleryBaseApplication {

	public static void main(String[] args) {
		SpringApplication.run(AigcGalleryBaseApplication.class, args);
	}

}
