package dao.service;

import com.baomidou.mybatisplus.extension.service.IService;
import dao.entity.Messages;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
public interface MessagesService extends IService<Messages> {
    @Transactional
    List<Messages> getCommentList(int groupId);
}
