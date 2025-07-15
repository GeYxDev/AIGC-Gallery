package dao.service;

import dao.entity.Images;
import com.baomidou.mybatisplus.extension.service.IService;
import org.springframework.transaction.annotation.Transactional;
import util.Feedback;

public interface ImagesService extends IService<Images> {
    @Transactional
    Feedback submitImage(Images images);
    @Transactional
    Feedback getImage(int workId);
    @Transactional
    Feedback deleteImage(int workId);
}
