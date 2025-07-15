package dao.service;

import dao.entity.Videos;
import com.baomidou.mybatisplus.extension.service.IService;
import org.springframework.transaction.annotation.Transactional;
import util.Feedback;

public interface VideosService extends IService<Videos> {
    @Transactional
    Feedback submitVideo(Videos videos);
    @Transactional
    Feedback getVideo(int workId);
    @Transactional
    Feedback deleteVideo(int workId);
}
