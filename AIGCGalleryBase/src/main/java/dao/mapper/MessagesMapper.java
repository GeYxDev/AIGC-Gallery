package dao.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import dao.entity.Messages;

import java.util.List;

public interface MessagesMapper extends BaseMapper<Messages> {
    List<Messages> selectCommentList(Integer groupId);
}
