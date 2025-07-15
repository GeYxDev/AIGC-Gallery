package dao.service;

import dao.entity.Works;
import com.baomidou.mybatisplus.extension.service.IService;
import org.springframework.transaction.annotation.Transactional;
import util.Feedback;

public interface WorksService extends IService<Works> {
    @Transactional
    Feedback submitWork(Works works);
    @Transactional
    Feedback getWorkByGroupId(int groupId);
    @Transactional
    Feedback getWorkByWorkId(int workId);
    @Transactional
    Feedback deleteWorkByGroupId(int groupId);
    @Transactional
    Feedback deleteWorkByWorkId(int workId);
}
