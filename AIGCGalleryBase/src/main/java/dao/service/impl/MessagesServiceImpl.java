package dao.service.impl;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import dao.entity.Messages;
import dao.mapper.MessagesMapper;
import dao.service.MessagesService;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class MessagesServiceImpl extends ServiceImpl<MessagesMapper, Messages> implements MessagesService {

    @Override
    public List<Messages> getCommentList(int groupId) {
        return getBaseMapper().selectCommentList(groupId);
    }
}
