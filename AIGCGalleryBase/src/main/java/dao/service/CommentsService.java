package dao.service;

import dao.entity.Comments;
import com.baomidou.mybatisplus.extension.service.IService;
import org.springframework.transaction.annotation.Transactional;
import util.Feedback;

public interface CommentsService extends IService<Comments> {
    @Transactional
    Feedback addComment(Comments comments);
    @Transactional
    Feedback loadCommentByGroupId(int groupId);
    @Transactional
    Feedback deleteCommentByGroupId(int groupId);
}
