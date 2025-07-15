package dao.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.github.houbb.sensitive.word.core.SensitiveWordHelper;
import dao.entity.Comments;
import dao.mapper.CommentsMapper;
import dao.service.CommentsService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.stereotype.Service;
import util.Feedback;

import java.util.List;

@Service
public class CommentsServiceImpl extends ServiceImpl<CommentsMapper, Comments> implements CommentsService {

    @Override
    public Feedback addComment(Comments comments) {
        try {
            // 身份验证由上级完成
            comments.setContent(SensitiveWordHelper.replace(comments.getContent()));
            boolean saveResult = save(comments);
            if (saveResult) {
                return new Feedback(true, "评论上传成功");
            } else {
                return new Feedback(false, "评论上传失败，数据库插入异常");
            }
        } catch (Exception e) {
            return new Feedback(false, "评论上传失败" + e.getMessage());
        }
    }

    @Override
    public Feedback loadCommentByGroupId(int groupId) {
        try {
            Comments storedComment = getOne(new LambdaQueryWrapper<Comments>().eq(Comments::getGroupId, groupId));
            if (storedComment == null) {
                return new Feedback(false, "评论不存在");
            } else {
                return new Feedback(true, storedComment);
            }
        } catch (Exception e) {
            return new Feedback(false, "评论加载失败" + e.getMessage());
        }
    }

    @Override
    public Feedback deleteCommentByGroupId(int groupId) {
        try {
            LambdaQueryWrapper<Comments> commentLambdaQueryWrapper = new LambdaQueryWrapper<>();
            commentLambdaQueryWrapper.eq(Comments::getGroupId, groupId);
            List<Comments> commentList = list(commentLambdaQueryWrapper);
            if (commentList.isEmpty()) {
                return new Feedback(true, "评论不存在");
            }
            boolean removeResult = remove(commentLambdaQueryWrapper);
            if (removeResult) {
                return new Feedback(true, "评论删除成功");
            } else {
                return new Feedback(false, "评论删除失败，数据库删除错误");
            }
        } catch (Exception e) {
            return new Feedback(false, "评论删除失败" + e.getMessage());
        }
    }
}
